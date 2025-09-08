import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import uuid, time
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.crud.github_user import get_user_id_from_github_id
from app.crud.mongo.code_chunks import delete_code_chunks
from app.crud.organization import get_organization_id_from_github_id
from app.crud.user_organization import get_organization_id_for_user_id
from app.database import get_mongo_db
from app.services.calculate_tokens import calculate_tokens
from app.services.check_code_file import is_code_file
from app.services.check_valid_pr_line_numbers import (
    clean_raw_patch,
    validate_comment_line_numbers,
)
from app.services.installation_token_service import fetch_installation_token_installid
from app.services.json_to_md_service import json_to_md_analysis, json_to_md_issue
from app.services.order_handler_service import sync_order_status
from app.services.pr_config_utils import parse_yaml_file
from app.services.pr_review_services.hybrid_line_number_service import (
    char_issues_linenum_ext,
)
from app.services.pr_review_services.pr_files_fetch_service import (
    parse_diff_to_file_objects,
)
from app.crud.analyses import insert_pr_analysis
from app.crud.users import get_user_by_id
from app.crud.analyses import insert_pr_analysis
from app.crud.usage import (
    get_tokens_usage_by_user_id_org_id,
    insert_characteristic_usage,
    insert_usage,
    upsert_tokens_usage_user_id_org_id,
)
from app.services.github_app_email_utils import send_usage_email
from app.services.pr_review_services.md_summary_service import (
    format_severity_characteristic_data,
)
from app.services.pr_review_services.pr_files_fetch_service import (
    fetch_new_version,
    trim_patch_before_hunks,
)
from app.api.pr_details import (
    get_pull_request_files,
    post_pull_request_comment,
    post_pull_request_line_comment,
    post_pull_request_status,
    delete_pull_request_comment,
    get_pull_request_diff,
)
from app.services.prompt_service import PromptService
from app.services.llm_endpoint_service import get_router_service
from app.dependencies import logger
from app.services.md_to_json_service import md_to_json
from app.services.language_ext_and_applicability_check import applicability_check
from app.models import (
    LLMUsage,
)
from app.config import (
    GITHUB_APP_ID,
    OPENAI_COST,
    PAID_TOKENS_LIMIT,
    SMALL_LINES_TOKEN_LIMIT,
    FREE_TOKENS_LIMIT,
    MAX_INPUT_TOKENS,
    FRONTEND_URL,
    YAML_FILE_PATH,
)
from app.services.rag_services.chunking_pipeline import (
    chunk_code_and_save_to_db,
    parse_code_and_extract_chunks,
)

model = "gpt-4o-mini"
temperature = 0.9


async def process_file(
    file,
    factor: str,
    prompt_service: PromptService,
    mongo_db: AsyncIOMotorDatabase,
    github_login: str,
    repo_name: str,
    pr_number: int = None,
    commit_id: str = None,
    file_patch=None,
    preferred_characteristics: list = None,
    additional_instructions: str = "",
):
    """
    Common file processing function that can be used for both PR and commit reviews.
    
    Args:
        file: File object with filename, status, new_content, patch
        factor: Analysis factor (e.g., 'power_analysis', 'owasp')
        prompt_service: PromptService instance
        mongo_db: MongoDB database instance
        github_login: GitHub username
        repo_name: Repository name
        pr_number: PR number (for PR reviews)
        commit_id: Commit hash (for commit reviews)
        file_patch: List of file patches (for PR reviews)
        preferred_characteristics: List of preferred characteristics (for PR reviews)
        additional_instructions: Additional instructions for analysis (for PR reviews)
    
    Returns:
        dict: Analysis results with file info, response, code language, and usage data
    """
    logger.info(f"Started processing file {file['filename']}")

    # Check for 'patch' in file object
    patch = file.get("patch")
    # Fallback: if patch is None or empty, look for it in file_patch list
    if not patch and file_patch:
        matched = next(
            (
                item.get("patch")
                for item in file_patch
                if item.get("filename") == file.get("filename")
            ),
            None,
        )
        patch = matched

    # If patch is still None or empty, exit early
    if not patch:
        logger.warning(
            f"No patch found for file {file.get('filename', 'unknown')}, skipping."
        )
        return

    req_usage_data = {}
    llm_service = get_router_service()
    try:
        cleaned_patch = clean_raw_patch(patch, file["filename"])
        if not cleaned_patch:
            return {
                "file": file,
                "response": "We are skipping the analysis for this file as there are no substantial changes.",
                "code_language": "",
            }
    except Exception as e:
        logger.error(
            f"Error occurred in cleaning raw patch: {str(e)}",
            extra={"patched_filename": file["filename"]},
        )
        cleaned_patch = patch

    is_jsx_tsx = 0
    extension = "." + file["filename"].split(".")[-1]
    if extension == ".jsx" or extension == ".tsx":
        is_jsx_tsx = 1
    else:
        is_jsx_tsx = 0

    # If the file is a JSX/TSX file, we can remove JSX elements
    # Uncomment the following lines if you want to enable JSX cleaning
    # This is currently disabled as per the original code

    # change here
    # try:
    #     cleaned_patch = remove_jsx_by_extension(cleaned_patch, file["filename"])

    #     if not cleaned_patch:
    #         return {
    #         "file": file,
    #         "response": "We are skipping the analysis for this file as there are no substantial changes.",
    #         "code_language": "",
    #     }

    # except Exception as e:
    #     logger.error(
    #     f"Error occurred in cleaning patch for jsx: {str(e)}",
    #     extra={"patched_filename": file["filename"]},
    # )
    #     cleaned_patch = patch

    code_tokens = calculate_tokens(cleaned_patch)

    if code_tokens < SMALL_LINES_TOKEN_LIMIT and factor in ["owasp", "soc2", "cwe"]:
        return {
            "file": file,
            "response": f"We are skipping the OWASP Top-10 analysis for this file as there are no substantial changes.",
            "code_language": "",
        }

    # Use appropriate context extraction based on whether it's PR or commit review
    if commit_id:
        context_chunks = await parse_code_and_extract_chunks(
            file["new_content"],
            mongo_db,
            github_login,
            file["filename"],
            repo_name,
            commit_id=commit_id,
        )
    else:
        context_chunks = await parse_code_and_extract_chunks(
            file["new_content"],
            mongo_db,
            github_login,
            file["filename"],
            repo_name,
            pr_number,
        )
    
    context = "\n".join(chunk["code_snippet"] for chunk in context_chunks)

    if file["status"] == "added":
        code_to_analyze = cleaned_patch
        code_context = context

    elif file["status"] == "modified":
        code_to_analyze = patch
        code_context = context + "\n" + file["new_content"]

    total_tokens = calculate_tokens(code_to_analyze) + calculate_tokens(code_context)
    if total_tokens > MAX_INPUT_TOKENS:
        logger.warning(
            f"We are skipping the analysis for this file as the total tokens are greater than the free tokens limit. {code_tokens} for file {file['filename']}"
        )
        return {}

    if is_jsx_tsx or (
        code_tokens <= SMALL_LINES_TOKEN_LIMIT and factor == "power_analysis"
    ):
        applicability_check_str = await prompt_service.get_prompt(
            "applicability_check_small_prompt", code=code_to_analyze, factor=factor
        )
    elif code_tokens > SMALL_LINES_TOKEN_LIMIT and factor not in ["cwe", "soc2"]:
        applicability_check_str = await prompt_service.get_prompt(
            "applicability_check_prompt", code=code_to_analyze, factor=factor
        )
    elif code_tokens > SMALL_LINES_TOKEN_LIMIT and factor in ["cwe", "soc2"]:
        applicability_check_str = await prompt_service.get_prompt(
            "applicability_check_prompt_cwe_soc2", code=code_to_analyze, factor=factor
        )

    else:
        logger.warning(
            f"Invalid factor {factor} and/or code length {code_tokens} for file {file['filename']}"
        )
        return {}

    applicability_check_prompt = [
        {
            "role": "system",
            "content": "You are a senior software engineer who is great at analyzing and reviewing code.",
        },
        {"role": "user", "content": applicability_check_str},
    ]
    applicability_usage_data = LLMUsage()
    applicability_check_response = {}
    llm_service = get_router_service()
    req_usage_data = {}

    try:
        applicability_check_response = await applicability_check(
            prompt=applicability_check_prompt,
            file_name=file["filename"],
            llm_service=llm_service,
            usage_data=applicability_usage_data,
        )

        cost = (
            applicability_usage_data.input_tokens
            / 1000
            * OPENAI_COST[model]["input_tokens"]
            + applicability_usage_data.response_tokens
            / 1000
            * OPENAI_COST[model]["response_tokens"]
        )
        applicability_usage_data.cost = cost
        req_usage_data["applicability_check_" + factor] = applicability_usage_data

        if code_tokens > SMALL_LINES_TOKEN_LIMIT:
            logger.info(
                "Filtered chars: %s",
                applicability_check_response.get("filtered_chars", []),
                extra={"patched_filename": file["filename"]},
            )

    except Exception as e:
        logger.error(
            "Error occurred while running applicability check for factor: %s, error: %s",
            factor,
            str(e),
        )

    if factor not in ["power_analysis", "owasp","cwe"]:
        prompts_dict = await prompt_service.get_prompt(
            "factor_analysis_prompt",
            factor_name=factor,
            applicable_chars=applicability_check_response.get("filtered_chars", []),
        )
    elif factor == "power_analysis":
        # Determine if is_jsx_tsx
        if is_jsx_tsx or code_tokens <= SMALL_LINES_TOKEN_LIMIT:
            logger.info("Running small file power analysis")
            prompts_dict = await prompt_service.get_prompt(
                "power_analysis_small_prompt",
                factor_name="power_analysis_small",
                context=code_context,
                additional_instructions=additional_instructions
            )
        else:
            filtered_chars = applicability_check_response.get("filtered_chars", [])
            if filtered_chars and preferred_characteristics:
                filtered_chars = [char for char in filtered_chars if char.lower() in preferred_characteristics]

            if not filtered_chars:
                return {
                    "file": file,
                    "response": f"We are skipping the analysis for this file as everything looks great.",
                    "code_language": "",
                }

            prompts_dict = await prompt_service.get_prompt(
                "power_analysis_prompt",
                factor_name=factor,
                context=code_context,
                applicable_chars=filtered_chars,
                additional_instructions=additional_instructions
            )

    elif factor == "owasp":
        prompts_dict = await prompt_service.get_prompt(
            "owasp_analysis_prompt",
            factor_name=factor,
            context=code_context,
            applicable_chars=applicability_check_response.get("filtered_chars", []),
        )
    
    elif factor == "cwe":
        filtered_chars = applicability_check_response.get("filtered_chars", [])
        if not filtered_chars:
            return {
                "file": file,
                "response": f"We are skipping the analysis for this file as everything looks great.",
                "code_language": "",
            }

        prompts_dict = await prompt_service.get_prompt(
            "cwe_analysis_prompt",
            factor_name=factor,
            context=code_context,
            applicable_chars=applicability_check_response.get("filtered_chars", []),
        )

    pr_review_response = []  # Collect responses for each characteristic

    for index, (characteristic, char_prompt) in enumerate(prompts_dict.items()):
        char_usage_data = LLMUsage()
        char_response = ""
        llm_char_response = ""

        prompt = [
            {
                "role": "system",
                "content": "You are a senior software engineer who is great at analyzing and reviewing code.",
            },
            {
                "role": "user",
                "content": f"Code To Analyze:\n{code_to_analyze}\n{char_prompt}",
            },
        ]

        try:
            if factor != "owasp":
                char_heading = f"# {characteristic}\n\n"
                char_response += char_heading
            # Generating markdown response
            async for chunk in llm_service.agenerate_streaming_response(
                prompt=prompt,
                model=model,
                usage_data=char_usage_data,
                temperature=temperature,
            ):
                llm_char_response += chunk

            char_response += f"{llm_char_response}\n\n"
            char_response_json = md_to_json(char_response, file["filename"], factor)

            file_code = file["new_content"]
            char_response_linenum = char_issues_linenum_ext(
                char_response_json, file_code
            )

            pr_review_response.extend(char_response_linenum)

            cost = (
                char_usage_data.input_tokens / 1000 * OPENAI_COST[model]["input_tokens"]
                + char_usage_data.response_tokens
                / 1000
                * OPENAI_COST[model]["response_tokens"]
            )
            char_usage_data.cost = cost
            req_usage_data[characteristic] = char_usage_data

        except Exception as e:
            logger.error(
                f"Error processing characteristic {characteristic}: {e}",
                extra={"patched_filename": file["filename"]},
            )

    # Calculate total tokens and total cost
    total_tokens = 0
    total_cost = 0.0

    for usage in req_usage_data.values():
        total_tokens += getattr(usage, "input_tokens", 0) + getattr(
            usage, "response_tokens", 0
        )
        total_cost += getattr(usage, "cost", 0.0)

    return {
        "file": file,
        "response": pr_review_response,
        "code_language": applicability_check_response.get("language", ""),
        "usage_summary": {"total_tokens": total_tokens, "total_cost": total_cost},
        "req_usage_data": req_usage_data,
    }


async def pr_review_pipeline(
    pr_details, repo_data, installation_id, factor, db_session
):
    try:
        pr_process_start_time = datetime.now(timezone.utc)
        repo_owner = repo_data.get("owner", {})
        github_username = pr_details.get("github_username")
        repo_owner_github_id = repo_data.get("owner", {}).get("id", None)
        repo_admin_username = repo_owner.get("login", "")

        # Get the stored GitHub installation token
        token = await fetch_installation_token_installid(
            db_session, str(installation_id)
        )
        github_id = pr_details.get("github_user_id")
        user_id = await get_user_id_from_github_id(db_session, int(github_id))
        organization_details = await get_organization_id_from_github_id(
            db_session=db_session,
            platform_type_id=1,
            platform_id=int(repo_owner_github_id),
        )
        matched_org_id = None
        if user_id is not None:
            user_id = str(user_id)
            user = await get_user_by_id(db_session, user_id)
            org_ids = await get_organization_id_for_user_id(
                db_session=db_session, user_id=user_id
            )
            for org_id, role in org_ids:
                if organization_details["id"] == org_id:
                    matched_org_id = org_id
                    break

        if user_id is None or matched_org_id is None:
            signup_link = f"{FRONTEND_URL}/signup"
            logger.warning(
                "user id is None or install_id is None",
                extra={"github_id": github_id, "install_id": installation_id},
            )
            await post_pull_request_comment(
                repo_owner["login"],
                repo_data["name"],
                pr_details["number"],
                f"We could not run your PR Review. We noticed that you are part of an Org. We require everyone who is part of an Org to SignUp via GitHub so we can track your individual usage and maximize on your usage capacity. Enroll into CodeSherlock system by signing up via GitHub using the [SignUp link]({signup_link}). Also, please note — every user pays for their own usage.",
                token["access_token"],
            )
            return

        try:
            order_details = await sync_order_status(
                db_session=db_session, user_id=user_id, org_id=matched_org_id
            )
            order_status = order_details.get("status")
            tokens_usage = await get_tokens_usage_by_user_id_org_id(
                db_session=db_session, user_id=user_id, organization_id=matched_org_id
            )
        except Exception as e:
            # Log the error for debugging, optionally send a user-friendly message
            logger.error(
                f"We ran into an issue while retrieving your usage details. {str(e)}",
                extra={"user_id": user_id, "error": str(e)},
            )
            await post_pull_request_comment(
                repo_owner["login"],
                repo_data["name"],
                pr_details["number"],
                "We ran into an issue while retrieving your usage details. Please try again later or contact support@codesherlock.ai.",
                token["access_token"],
            )
            return

        # Set token limit based on the order status
        if order_status == "active":
            token_limit = PAID_TOKENS_LIMIT
        else:
            token_limit = FREE_TOKENS_LIMIT

        initial_message = ""
        tokens_left = max(0, token_limit - tokens_usage)
        percentage_remaining = max(0, int((tokens_left / token_limit) * 100))

        if not user.role and percentage_remaining <= 40:
            dashboard_link = f"{FRONTEND_URL}/dashboard"
            initial_message = (
                f"⚠️ **Your Codesherlock Usage!**\n\n"
                f"You're down to {tokens_left} tokens, (**{percentage_remaining}%**) of your free allowance.\n\n"
                f"Once they’re gone, new analysis in‑editor and PR analyses will pause.\n\n"
                f"You can upgrade your to paid plan from here: [View Dashboard]({dashboard_link})."
            )

        # Check based on order status and tokens_usage
        if not user.role:
            if tokens_left == 0:
                await post_pull_request_comment(
                    repo_owner["login"],
                    repo_data["name"],
                    pr_details["number"],
                    (
                        f"⚠️ **Your Codesherlock Usage!**\n\n"
                        f"You're down to {tokens_left} tokens, (**{percentage_remaining}%**) of your free allowance.\n\n"
                        f"To continue using CodeSherlock's analysis, please  upgrade to a monthly subscription.\n\n"
                        f"You can upgrade your to paid plan from here: [View Dashboard]({dashboard_link})."
                    ),
                    token["access_token"],
                )
                return

        pr_reviewers = [
            reviewer.get("login")
            for reviewer in pr_details.get("requested_reviewers", [])
        ]

        pr_details["reviewers"] = pr_reviewers

        codesherlock_config_file = {}
        codesherlock_config_file["filename"] = YAML_FILE_PATH
        file_content = await fetch_new_version(
            owner=repo_owner["login"],
            repo=repo_data["name"],
            filename=codesherlock_config_file["filename"],
            head_sha=pr_details["head_sha"],
            access_token=token["access_token"]
        )
        codesherlock_config_file["new_content"] = file_content["new_content"]
        pr_config = {}
        if codesherlock_config_file["new_content"]:
            try:
                pr_config = await parse_yaml_file(codesherlock_config_file["new_content"])
            except Exception as e:
                logger.error(
                    f"Error parsing codesherlock.yaml file: {str(e)}",
                    extra={"owner": repo_owner["login"], "pr_number": pr_details["number"]},)

        if pr_config.get("target_branches") and pr_details.get("base_branch") not in pr_config.get("target_branches"):
            logger.info(
                f"Skipping PR {pr_details['number']} because branch {pr_details.get('base_branch')} is not in the target branches {pr_config.get('target_branches')}",
                extra={"owner": repo_owner["login"], "pr_number": pr_details["number"]},
            )

            await post_pull_request_status(
                repo_owner["login"],
                repo_data["name"],
                commit_sha=pr_details["head_sha"],
                state="success",
                description="CodeSherlock.AI has skipped the review.",
                access_token=token["access_token"],
            )
            return

        initial_comment_task = post_pull_request_comment(
            repo_owner["login"],
            repo_data["name"],
            pr_details["number"],
            "**CodeSherlock.AI** is currently reviewing the changes in this pull request.\n\n⏳ *Smaller PRs typically take 1–2 minutes, medium ones 3–4 minutes, and larger PRs may take up to 5–6 minutes.*",
            token["access_token"],
        )

        owasp_tip = (
            "### 💡 Tip\n"
            "Want to run a security-focused check?  \n"
            f"Comment `@{GITHUB_APP_ID} analyze owasp` on this PR to trigger an **OWASP Top-10 security analysis**.\n\n"
        )

        owasp_tip_task = post_pull_request_comment(
            repo_owner["login"],
            repo_data["name"],
            pr_details["number"],
            owasp_tip,
            token["access_token"],
        )

        status_task = post_pull_request_status(
            repo_owner["login"],
            repo_data["name"],
            commit_sha=pr_details["head_sha"],
            state="pending",
            description="CodeSherlock.AI is reviewing this PR... ",
            access_token=token["access_token"],
        )

        file_details_task = get_pull_request_files(
            repo_owner["login"],
            repo_data["name"],
            pr_details["number"],
            token["access_token"],
        )

        tasks = [
            initial_comment_task,
            owasp_tip_task,
            status_task,
            file_details_task,
        ]

        if initial_message:
            tasks.append(
                post_pull_request_comment(
                    repo_owner["login"],
                    repo_data["name"],
                    pr_details["number"],
                    initial_message,
                    token["access_token"],
                )
            )

        initial_comment_response, _, _, pr_file_details, *rest = await asyncio.gather(
            *tasks
        )

        file_patch = []

        # Check if any files are missing patches
        missing_patches = any(
            file.get("changes", 0) > 0 and not file.get("patch")
            for file in pr_file_details
        )

        if missing_patches:
            start_time = time.time()
            diff_patch = await get_pull_request_diff(
                repo_owner["login"],
                repo_data["name"],
                pr_details["number"],
                token["access_token"],
            )
            elapsed_time = time.time() - start_time
            logger.info(
                f"Time elapsed for get_pull_request_diff: {elapsed_time:.2f} seconds"
            )

            file_patch = parse_diff_to_file_objects(diff_patch)

            # Step 3: convert list to filename -> patch string map
            file_patch_map = {}

            for f in file_patch:
                raw_patch = f.get("patch")
                if not raw_patch:
                    continue

                # Clean and trim the patch
                patch = raw_patch.replace("\r", "").strip()
                patch = trim_patch_before_hunks(patch)

                file_patch_map[f["filename"]] = patch

            # Step 4: update missing patches
            for file in pr_file_details:
                if not file.get("patch") and file_patch_map.get(file["filename"]):
                    file["patch"] = file_patch_map[file["filename"]]

        initial_comment_id = initial_comment_response["id"]

        logger.info(
            "Starting to analyze files",
            extra={"owner": repo_owner["login"], "pr_number": pr_details["number"]},
        )

        relevant_files = [
            file
            for file in pr_file_details
            if is_code_file(file["filename"])
            and file["status"] in ("added", "modified")
        ]

        # First, create a mapping of files to their fetch task
        fetch_tasks = [
            fetch_new_version(
                repo_owner["login"],
                repo_data["name"],
                file["filename"],
                pr_details["head_sha"],
                token["access_token"]
            )
            for file in relevant_files
        ]

        # Run all fetches concurrently
        new_contents = await asyncio.gather(*fetch_tasks)

        # Append the results back to the file objects
        for file, new_content in zip(relevant_files, new_contents):
            file["new_content"] = new_content["new_content"]

        prompt_service = PromptService()

        preferred_characteristics = pr_config.get("preferred_characteristics", [])

        additional_instructions = pr_config.get("additional_instructions", "")

        mongo_db = get_mongo_db()
        start_time = datetime.now()
        # Chunk code files and save to db
        try:
            for file in relevant_files:
                chunk_ids = await chunk_code_and_save_to_db(
                    file["new_content"],
                    mongo_db,
                    repo_owner["login"],
                    file["filename"],
                    repo_data["name"],
                    pr_details["number"],
                )
                if not chunk_ids:
                    logger.warning(f"No chunks inserted for file {file['filename']}")

        except Exception as e:
            logger.error(f"Error occured while creating context {e}")

        end_time = datetime.now()
        logger.info(f"Time taken to create context: {end_time - start_time}")

        # Process each file with its corresponding old version
        results = await asyncio.gather(
            *[
                process_file(
                    file=file,
                    factor=factor,
                    prompt_service=prompt_service,
                    mongo_db=mongo_db,
                    github_login=repo_owner["login"],
                    repo_name=repo_data["name"],
                    pr_number=pr_details["number"],
                    file_patch=file_patch,
                    preferred_characteristics=preferred_characteristics,
                    additional_instructions=additional_instructions
                )
                for file in relevant_files
            ]
        )

        pr_review_summary = ""
        pr_review_full = ""
        pr_review_line_comments = []

        logger.info(
            "All files analyzed. Constructing markdown..",
            extra={"sender": github_username, "pr_number": pr_details["number"]},
        )
        try:
            for result in results:
                if result:
                    response = result.get("response", [])
                    file = result.get("file", {})
                    code_language = result["code_language"]

                    # Constructing the Full Review markdown
                    if response:
                        pr_review_summary += f"## File Name: {file['filename']}\n\n"
                        pr_review_full += f"## File Name: {file['filename']}\n\n"
                        if type(response) == str:
                            pr_review_summary += f"{response}\n\n"
                            pr_review_full += f"{response}\n\n"
                            continue

                        # file_markdown = json_to_md_analysis(response, code_language)
                        # if file_markdown:
                        #     pr_review_full += (
                        #         f"## File Name: {file['filename']}\n\n{file_markdown}\n\n"
                        #     )

                        file_characteristic_data = defaultdict(dict)
                        valid_response = []

                        # Constructing line comments markdown
                        file_line_comments = {
                            "filename": file["filename"],
                            "line_comments": [],
                        }
                        for char_el in response:
                            if not char_el:
                                continue

                            issue_items = char_el.get("issue_items", [])
                            valid_issue_items = []

                            for issue_item in issue_items:
                                issue_markdown = json_to_md_issue(
                                    issue_item,
                                    code_language,
                                    char_el.get("characteristic"),
                                )
                                if not issue_markdown:
                                    continue

                                line_comment = {
                                    "start_line": issue_item.get("start_line"),
                                    "end_line": issue_item.get("end_line"),
                                    "comment": issue_markdown,
                                    "valid": True,
                                }
                                if line_comment.get("start_line"):
                                    file_line_comments["line_comments"].append(
                                        line_comment
                                    )

                                    validate_comment_line_numbers(
                                        file["patch"],
                                        file["filename"],
                                        line_comment,
                                    )
                                    if line_comment["valid"]:
                                        valid_issue_items.append(issue_item)

                                        file_characteristic_data[
                                            char_el["characteristic"]
                                        ][issue_item["severity"]] = (
                                            file_characteristic_data[
                                                char_el["characteristic"]
                                            ].get(issue_item["severity"], 0)
                                            + 1
                                        )

                            if valid_issue_items:
                                valid_char_el = {
                                    **char_el,
                                    "issue_items": valid_issue_items,
                                }
                                valid_response.append(valid_char_el)

                        pr_review_line_comments.append(file_line_comments)
                        pr_review_summary += await format_severity_characteristic_data(
                            file_characteristic_data
                        )
                        pr_review_full += json_to_md_analysis(valid_response, code_language)

        except Exception as e:
            logger.error(
                f"Error occured while md construction:{str(e)}",
                extra={"sender": github_username, "pr_number": pr_details["number"]},
            )

        if pr_review_summary:
            logger.info(
                "Final response markdown generated and posting...",
                extra={"sender": github_username, "pr_number": pr_details["number"]},
            )
            await post_pull_request_comment(
                repo_owner["login"],
                repo_data["name"],
                pr_details["number"],
                pr_review_summary,
                token["access_token"],
            )

        for file_line_comment in pr_review_line_comments:
            try:
                filename = file_line_comment["filename"]
                line_comments = file_line_comment["line_comments"]
                file = next(
                    (f for f in relevant_files if f["filename"] == filename), None
                )
                # Check for 'patch' in file object
                patch = file.get("patch")
                # Fallback: if patch is None or empty, look for it in file_patch list
                if not patch and file_patch:
                    matched = next(
                        (
                            item.get("patch")
                            for item in file_patch
                            if item.get("filename") == file.get("filename")
                        ),
                        None,
                    )
                    patch = matched
                patch = patch.replace("\r", "").strip()
                patch = trim_patch_before_hunks(patch)

            except Exception as e:
                logger.error(
                    f"Error occured while validating line numbers: {str(e)}",
                    extra={
                        "sender": github_username,
                        "pr_number": pr_details["number"],
                    },
                )
                continue

            for line_comment in line_comments:
                if line_comment["valid"]:
                    logger.info(
                        f"Posting comment at {line_comment['start_line']}-{line_comment['end_line']} in file {filename}..."
                    )
                    try:
                        await post_pull_request_line_comment(
                            owner=repo_owner["login"],
                            repo=repo_data["name"],
                            pr_number=pr_details["number"],
                            body=line_comment["comment"],
                            commit_id=pr_details["head_sha"],
                            path=filename,
                            start_line=line_comment["start_line"],
                            end_line=line_comment["end_line"],
                            access_token=token["access_token"],
                        )
                    except Exception as e:
                        logger.error(
                            f"Error occurred while posting line comment at lines {line_comment['start_line']}-{line_comment['end_line']}: {str(e)}"
                        )

        if pr_details["reviewers"]:
            reviewer_comment = "**CodeSherlock.AI** has completed its review. ✅"
            for reviewer in pr_details["reviewers"]:
                reviewer_comment = f"@{reviewer} {reviewer_comment}"

            logger.info("Posting Reviewer comment")
            await post_pull_request_comment(
                repo_owner["login"],
                repo_data["name"],
                pr_details["number"],
                reviewer_comment,
                token["access_token"],
            )

        await post_pull_request_status(
            repo_owner["login"],
            repo_data["name"],
            commit_sha=pr_details["head_sha"],
            state="success",
            description="CodeSherlock.AI has completed the review. PR is ready for merge.",
            access_token=token["access_token"],
        )

        await delete_pull_request_comment(
            repo_owner["login"],
            repo_data["name"],
            initial_comment_id,
            token["access_token"],
        )

        if pr_review_full:

            # ✅ Save PR analysis here immediately
            try:
                analysisid = str(uuid.uuid4())  # Generate a UUID string
                await insert_pr_analysis(
                    db_session=db_session,
                    user_id=user_id,
                    content=pr_review_full,
                    factor=factor,
                    language=results[0].get("code_language", "unknown"),
                    pr_number=pr_details["number"],
                    repo_name=repo_data["name"],
                    repo_owner=repo_admin_username,
                    analysisid=analysisid,
                )
            except Exception as e:
                logger.error(
                    f"Error while saving PR analysis: {str(e)}",
                    extra={
                        "sender": github_username,
                        "pr_number": pr_details["number"],
                    },
                )

        # Calculate total token usage and cost
        total_tokens_used = sum(
            result["usage_summary"]["total_tokens"]
            for result in results
            if "usage_summary" in result
        )
        total_cost = sum(
            result["usage_summary"]["total_cost"]
            for result in results
            if "usage_summary" in result
        )

        # Call the upsert function to record token usage
        await upsert_tokens_usage_user_id_org_id(
            db_session=db_session,
            user_id=user_id,
            organization_id=matched_org_id,
            tokens_used=total_tokens_used,
            cost=total_cost,
        )
        created_at = datetime.now(timezone.utc)

        for result in results:
            req_usage_data = result.get("req_usage_data", None)
            if req_usage_data:
                usage_ids = await insert_usage(
                    db_session=db_session,
                    usage_data=req_usage_data,
                    user_id=user_id,
                    model=model,
                    created_at=created_at,
                    organization_id=matched_org_id,
                )

                await insert_characteristic_usage(
                    db_session=db_session,
                    usage_data=req_usage_data,
                    usage_ids=usage_ids,
                    user_id=user_id,
                    created_at=created_at,
                    organization_id=matched_org_id,
                )

        await db_session.commit()

        tokens_usage_after_analysis = await get_tokens_usage_by_user_id_org_id(
            db_session, user_id, matched_org_id
        )

        tokens_left_after_analysis = max(0, token_limit - tokens_usage_after_analysis)
        percentage_remaining_after_analysis = max(
            0, int((tokens_left_after_analysis / token_limit) * 100)
        )

        if not user.role:
            thresholds = [40, 20, 10, 0]
            crossed_thresholds = [
                t
                for t in thresholds
                if percentage_remaining >= t >= percentage_remaining_after_analysis
            ]

            if crossed_thresholds:
                await send_usage_email(
                    user.email,
                    username=github_username,
                    tokens_left=tokens_left_after_analysis,
                    percentage_remaining=percentage_remaining_after_analysis,
                )

        delete_count = await delete_code_chunks(
            mongo_db, pr_details["number"], github_username, repo_data["name"]
        )
        logger.info(
            f"{delete_count} code chunks deleted for PR: {pr_details['number']}",
            extra={"owner": repo_owner["login"], "pr_number": pr_details["number"]},
        )

        total_complete_time = datetime.now(timezone.utc)
        logger.info(
            f"PR analysis completed in {total_complete_time - pr_process_start_time}",
            extra={"user_id": user_id},
        )

        return results

    except Exception as e:
        logger.error(
            f"Error occurred while processing PR: {str(e)}",
            extra={"user_id": user_id, "org_id": matched_org_id},
        )