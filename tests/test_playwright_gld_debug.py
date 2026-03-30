import os
import sys
import unittest
from datetime import datetime, timedelta


try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


class TestGLDFlowDebug(unittest.TestCase):
    """UI smoke + debug test for GLD analysis flow."""

    @unittest.skipIf(sync_playwright is None, "playwright is not installed")
    def test_gld_analysis_flow_no_error_dialog(self):
        base_url = os.environ.get("QUANTAGENT_BASE_URL", "http://127.0.0.1:5000")
        selected_timeframe = os.environ.get("QUANTAGENT_TEST_TIMEFRAME", "5m")
        selected_asset = os.environ.get("QUANTAGENT_TEST_ASSET", "GLD")
        timeout_ms = int(os.environ.get("QUANTAGENT_TEST_TIMEOUT_MS", "90000"))

        # Keep dates deterministic but configurable.
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)
        start_date = os.environ.get("QUANTAGENT_TEST_START_DATE", start_dt.strftime("%Y-%m-%d"))
        end_date = os.environ.get("QUANTAGENT_TEST_END_DATE", end_dt.strftime("%Y-%m-%d"))
        start_time = os.environ.get("QUANTAGENT_TEST_START_TIME", "00:00")
        end_time = os.environ.get("QUANTAGENT_TEST_END_TIME", "23:59")

        debug_lines = []
        dialog_messages = []
        response_errors = []

        def log(msg: str) -> None:
            line = f"[pw-debug] {msg}"
            debug_lines.append(line)
            print(line, flush=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            page.on("console", lambda m: log(f"console.{m.type}: {m.text}"))
            page.on("pageerror", lambda e: log(f"pageerror: {e}"))

            def on_response(resp):
                if "/api/" in resp.url:
                    log(f"response {resp.status} {resp.url}")
                if "/api/analyze" in resp.url and resp.status >= 400:
                    response_errors.append(f"{resp.status} {resp.url}")

            page.on("response", on_response)

            def on_dialog(dialog):
                msg = dialog.message
                dialog_messages.append(msg)
                log(f"dialog.{dialog.type}: {msg}")
                dialog.dismiss()

            page.on("dialog", on_dialog)

            log(f"navigating to {base_url}/demo")
            page.goto(f"{base_url}/demo", wait_until="networkidle", timeout=timeout_ms)

            # Select asset.
            asset_btn = page.locator(f".asset-btn[data-asset='{selected_asset}']")
            self.assertTrue(asset_btn.count() > 0, f"Asset button not found: {selected_asset}")
            asset_btn.first.click()
            log(f"selected asset={selected_asset}")

            # Select timeframe.
            tf_btn = page.locator(f".timeframe-btn[data-timeframe='{selected_timeframe}']")
            self.assertTrue(
                tf_btn.count() > 0, f"Timeframe button not found: {selected_timeframe}"
            )
            tf_btn.first.click()
            log(f"selected timeframe={selected_timeframe}")

            # Set explicit date/time inputs.
            page.fill("#startDate", start_date)
            page.fill("#endDate", end_date)
            page.fill("#startTime", start_time)
            page.fill("#endTime", end_time)
            log(
                f"date range start={start_date} {start_time}, end={end_date} {end_time}"
            )

            analyze_btn = page.locator("#analyzeBtn")
            self.assertTrue(analyze_btn.count() > 0, "Analyze button not found")

            # Click analyze and wait for backend call.
            with page.expect_response(
                lambda r: "/api/analyze" in r.url, timeout=timeout_ms
            ) as analyze_resp_info:
                analyze_btn.click()

            analyze_resp = analyze_resp_info.value
            body_text = analyze_resp.text()
            log(f"/api/analyze status={analyze_resp.status}")
            log(f"/api/analyze body={body_text[:1200]}")

            # Grab visible status text if present.
            status_text = ""
            status_locator = page.locator("#apiKeyStatusText")
            if status_locator.count() > 0:
                status_text = status_locator.first.inner_text().strip()
                if status_text:
                    log(f"status text={status_text}")

            # Snapshot for post-mortem.
            screenshot_path = "playwright_debug_failure.png"
            page.screenshot(path=screenshot_path, full_page=True)
            log(f"saved screenshot={screenshot_path}")

            browser.close()

        # Fail loudly with full debug context.
        if dialog_messages:
            self.fail(
                "Unexpected UI dialog(s):\n"
                + "\n".join(dialog_messages)
                + "\n\nDebug log:\n"
                + "\n".join(debug_lines)
            )
        if response_errors:
            self.fail(
                "HTTP error responses from API:\n"
                + "\n".join(response_errors)
                + "\n\nDebug log:\n"
                + "\n".join(debug_lines)
            )


if __name__ == "__main__":
    if sync_playwright is None:
        print("Playwright missing. Install with: pip install playwright && playwright install", flush=True)
        sys.exit(1)
    unittest.main(verbosity=2)
