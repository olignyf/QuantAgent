import json
import os
import re
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, send_file
from openai import OpenAI

import static_util
from trading_graph import TradingGraph

app = Flask(__name__)


class WebTradingAnalyzer:
    def __init__(self):
        """Initialize the web trading analyzer."""
        from default_config import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG.copy()
        self.trading_graph = TradingGraph(config=self.config)
        self.data_dir = Path("data")
        self.cache_dir = Path("cache")

        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Available assets and their display names
        self.asset_mapping = {
            "SPX": "S&P 500",
            "BTC": "Bitcoin",
            "GC": "Gold Futures",
            "NQ": "Nasdaq Futures",
            "CL": "Crude Oil",
            "ES": "E-mini S&P 500",
            "DJI": "Dow Jones",
            "QQQ": "Invesco QQQ Trust",
            "VIX": "Volatility Index",
            "DXY": "US Dollar Index",
            "AAPL": "Apple Inc.",  # New asset
            "TSLA": "Tesla Inc.",  # New asset
        }

        # Yahoo Finance symbol mapping
        self.yfinance_symbols = {
            "SPX": "^GSPC",  # S&P 500
            "BTC": "BTC-USD",  # Bitcoin
            "GC": "GC=F",  # Gold Futures
            "NQ": "NQ=F",  # Nasdaq Futures
            "CL": "CL=F",  # Crude Oil
            "ES": "ES=F",  # E-mini S&P 500
            "DJI": "^DJI",  # Dow Jones
            "QQQ": "QQQ",  # Invesco QQQ Trust
            "VIX": "^VIX",  # Volatility Index
            "DXY": "DX-Y.NYB",  # US Dollar Index
        }

        # Yahoo Finance interval mapping
        self.yfinance_intervals = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",  # yfinance supports 4h natively!
            "1d": "1d",
            "1w": "1wk",
            "1mo": "1mo",
        }

        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()

    def fetch_yfinance_data(
        self, symbol: str, interval: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            df = yf.download(
                tickers=yf_symbol, start=start_date, end=end_date, interval=yf_interval
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Ensure df is a DataFrame, not a Series
            if isinstance(df, pd.Series):
                df = df.to_frame()

            # Reset index to ensure we have a clean DataFrame
            df = df.reset_index()

            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()

            # Handle potential MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns if needed
            column_mapping = {
                "Date": "Datetime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }

            # Only rename columns that exist
            existing_columns = {
                old: new for old, new in column_mapping.items() if old in df.columns
            }
            df = df.rename(columns=existing_columns)

            # Ensure we have the required columns
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Missing columns. Available: {list(df.columns)}")
                return pd.DataFrame()

            # Select only the required columns
            df = df[required_columns]
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _normalize_yfinance_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output into a consistent OHLC DataFrame."""
        empty_df = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close"])
        if df is None or df.empty:
            return empty_df

        if isinstance(df, pd.Series):
            df = df.to_frame()

        df = df.reset_index()

        if not isinstance(df, pd.DataFrame):
            return empty_df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        column_mapping = {
            "Date": "Datetime",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
        existing_columns = {old: new for old, new in column_mapping.items() if old in df.columns}
        df = df.rename(columns=existing_columns)

        required_columns = ["Datetime", "Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing columns. Available: {list(df.columns)}")
            return empty_df

        df = df[required_columns].copy()
        df["Datetime"] = self._to_naive_datetime_series(df["Datetime"])
        df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).reset_index(drop=True)
        return df

    def _to_naive_datetime_series(self, dt_series: pd.Series) -> pd.Series:
        """Convert timezone-aware datetime values into naive datetimes consistently."""
        parsed = pd.to_datetime(dt_series, errors="coerce")
        # pd.to_datetime may return either Series or DatetimeIndex depending on input type.
        if isinstance(parsed, pd.DatetimeIndex):
            if parsed.tz is not None:
                parsed = parsed.tz_convert("UTC").tz_localize(None)
            return pd.Series(parsed)

        try:
            if hasattr(parsed, "dt") and getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
            return parsed
        except Exception:
            # Fallback for mixed values; normalize element-wise.
            normalized = pd.Series(parsed).apply(
                lambda x: (
                    x.tz_convert("UTC").tz_localize(None)
                    if pd.notna(x) and getattr(x, "tzinfo", None) is not None
                    else x
                )
            )
            return pd.to_datetime(normalized, errors="coerce")

    def _to_naive_datetime(self, dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            return dt.astimezone().replace(tzinfo=None)
        return dt

    def _cache_key_symbol(self, symbol: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)

    def _cache_file_path(self, symbol: str, interval: str, day: date) -> Path:
        safe_symbol = self._cache_key_symbol(symbol)
        return self.cache_dir / f"{safe_symbol}_{interval}_{day.isoformat()}.csv"

    def _cache_meta_path(self, symbol: str, interval: str, day: date) -> Path:
        safe_symbol = self._cache_key_symbol(symbol)
        return self.cache_dir / f"{safe_symbol}_{interval}_{day.isoformat()}.meta.json"

    def _read_day_cache(self, symbol: str, interval: str, day: date) -> pd.DataFrame:
        empty_df = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close"])
        path = self._cache_file_path(symbol, interval, day)
        if not path.exists():
            return empty_df
        try:
            df = pd.read_csv(path)
            if df.empty:
                return empty_df
            if "Datetime" not in df.columns:
                return empty_df
            df["Datetime"] = self._to_naive_datetime_series(df["Datetime"])
            return (
                df[["Datetime", "Open", "High", "Low", "Close"]]
                .sort_values("Datetime")
                .drop_duplicates(subset=["Datetime"])
                .reset_index(drop=True)
            )
        except Exception as e:
            print(f"Error reading cache file {path}: {e}")
            return empty_df

    def _write_day_cache(self, symbol: str, interval: str, day: date, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        path = self._cache_file_path(symbol, interval, day)
        out = (
            df[["Datetime", "Open", "High", "Low", "Close"]]
            .copy()
            .sort_values("Datetime")
            .drop_duplicates(subset=["Datetime"])
        )
        out.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    def _load_covered_ranges(self, symbol: str, interval: str, day: date) -> list[tuple[datetime, datetime]]:
        meta_path = self._cache_meta_path(symbol, interval, day)
        if not meta_path.exists():
            return []
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_ranges = data.get("covered_ranges", [])
            ranges = []
            for item in raw_ranges:
                if not isinstance(item, list) or len(item) != 2:
                    continue
                start_dt = self._to_naive_datetime(datetime.fromisoformat(item[0]))
                end_dt = self._to_naive_datetime(datetime.fromisoformat(item[1]))
                if end_dt > start_dt:
                    ranges.append((start_dt, end_dt))
            return self._merge_ranges(ranges)
        except Exception as e:
            print(f"Error reading cache metadata {meta_path}: {e}")
            return []

    def _save_covered_ranges(
        self, symbol: str, interval: str, day: date, ranges: list[tuple[datetime, datetime]]
    ) -> None:
        meta_path = self._cache_meta_path(symbol, interval, day)
        merged = self._merge_ranges(ranges)
        payload = {
            "symbol": symbol,
            "interval": interval,
            "day": day.isoformat(),
            "covered_ranges": [[s.isoformat(), e.isoformat()] for s, e in merged],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _merge_ranges(self, ranges: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
        if not ranges:
            return []
        normalized = sorted(ranges, key=lambda x: x[0])
        merged = [normalized[0]]
        for start_dt, end_dt in normalized[1:]:
            last_start, last_end = merged[-1]
            if start_dt <= last_end:
                merged[-1] = (last_start, max(last_end, end_dt))
            else:
                merged.append((start_dt, end_dt))
        return merged

    def _missing_ranges(
        self,
        requested_start: datetime,
        requested_end: datetime,
        covered_ranges: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        if requested_end <= requested_start:
            return []
        if not covered_ranges:
            return [(requested_start, requested_end)]

        merged = self._merge_ranges(covered_ranges)
        missing = []
        cursor = requested_start
        for cov_start, cov_end in merged:
            if cov_end <= cursor:
                continue
            if cov_start > cursor:
                missing.append((cursor, min(cov_start, requested_end)))
            cursor = max(cursor, cov_end)
            if cursor >= requested_end:
                break
        if cursor < requested_end:
            missing.append((cursor, requested_end))
        return [(s, e) for s, e in missing if e > s]

    def _download_yfinance_range(
        self, yf_symbol: str, yf_interval: str, start_datetime: datetime, end_datetime: datetime
    ) -> pd.DataFrame:
        df = yf.download(
            tickers=yf_symbol,
            start=start_datetime,
            end=end_datetime,
            interval=yf_interval,
            auto_adjust=True,
            prepost=True,
        )
        return self._normalize_yfinance_df(df)

    def fetch_yfinance_data_with_datetime(
        self,
        symbol: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance with day-sliced local cache support."""
        try:
            start_datetime = self._to_naive_datetime(start_datetime)
            end_datetime = self._to_naive_datetime(end_datetime)
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            print(
                f"Fetching {yf_symbol} from {start_datetime} to {end_datetime} with interval {yf_interval}"
            )
            all_frames = []
            start_day = start_datetime.date()
            end_day = end_datetime.date()

            day_cursor = start_day
            while day_cursor <= end_day:
                day_start = datetime.combine(day_cursor, datetime.min.time())
                next_day_start = day_start + timedelta(days=1)
                request_start = max(start_datetime, day_start)
                request_end = min(end_datetime, next_day_start)

                if request_end <= request_start:
                    day_cursor += timedelta(days=1)
                    continue

                cached_df = self._read_day_cache(yf_symbol, yf_interval, day_cursor)
                covered_ranges = self._load_covered_ranges(yf_symbol, yf_interval, day_cursor)
                missing_ranges = self._missing_ranges(request_start, request_end, covered_ranges)
                print(
                    f"[cache-debug] {yf_symbol} {yf_interval} {day_cursor}: "
                    f"cached_rows={len(cached_df)} covered={len(covered_ranges)} missing={len(missing_ranges)} "
                    f"window={request_start} -> {request_end}"
                )

                for missing_start, missing_end in missing_ranges:
                    fetched_df = self._download_yfinance_range(
                        yf_symbol=yf_symbol,
                        yf_interval=yf_interval,
                        start_datetime=missing_start,
                        end_datetime=missing_end,
                    )
                    if fetched_df is not None and not fetched_df.empty:
                        print(
                            f"[cache-debug] fetched {len(fetched_df)} rows for {day_cursor} "
                            f"range {missing_start} -> {missing_end}, "
                            f"rows {fetched_df['Datetime'].min()} -> {fetched_df['Datetime'].max()}"
                        )
                        cached_df = pd.concat([cached_df, fetched_df], ignore_index=True)
                        cached_df = (
                            cached_df.sort_values("Datetime")
                            .drop_duplicates(subset=["Datetime"])
                            .reset_index(drop=True)
                        )
                    else:
                        print(
                            f"[cache-debug] fetched 0 rows for {day_cursor} "
                            f"range {missing_start} -> {missing_end}"
                        )
                    covered_ranges.append((missing_start, missing_end))

                if cached_df is not None and not cached_df.empty:
                    self._write_day_cache(yf_symbol, yf_interval, day_cursor, cached_df)
                self._save_covered_ranges(yf_symbol, yf_interval, day_cursor, covered_ranges)

                if "Datetime" in cached_df.columns and not cached_df.empty:
                    day_slice = cached_df[
                        (cached_df["Datetime"] >= request_start) & (cached_df["Datetime"] < request_end)
                    ]
                    if not day_slice.empty:
                        all_frames.append(day_slice)

                day_cursor += timedelta(days=1)

            if not all_frames:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()

            merged = (
                pd.concat(all_frames, ignore_index=True)
                .sort_values("Datetime")
                .drop_duplicates(subset=["Datetime"])
                .reset_index(drop=True)
            )
            print(f"Successfully loaded {len(merged)} data points for {symbol}")
            print(f"Date range: {merged['Datetime'].min()} to {merged['Datetime'].max()}")
            return merged

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_available_assets(self) -> list:
        """Get list of available assets from the asset mapping dictionary."""
        return sorted(list(self.asset_mapping.keys()))

    def get_available_files(self, asset: str, timeframe: str) -> list:
        """Get available data files for a specific asset and timeframe."""
        asset_dir = self.data_dir / asset.lower()
        if not asset_dir.exists():
            return []

        pattern = f"{asset}_{timeframe}_*.csv"
        files = list(asset_dir.glob(pattern))
        return sorted(files)

    def run_analysis(
        self, df: pd.DataFrame, asset_name: str, timeframe: str
    ) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            # Debug: Check DataFrame structure
            print(f"DataFrame columns: {df.columns}")
            print(f"DataFrame index: {type(df.index)}")
            print(f"DataFrame shape: {df.shape}")

            # Prepare data for analysis
            # if len(df) > 49:
            #     df_slice = df.tail(49).iloc[:-3]
            # else:
            #     df_slice = df.tail(45)

            df_slice = df.tail(45)

            # Ensure DataFrame has the expected structure
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df_slice.columns for col in required_columns):
                return {
                    "success": False,
                    "error": f"Missing required columns. Available: {list(df_slice.columns)}",
                }

            # Reset index to avoid any MultiIndex issues
            df_slice = df_slice.reset_index(drop=True)

            # Debug: Check the slice before conversion
            print(f"Slice columns: {df_slice.columns}")
            print(f"Slice index: {type(df_slice.index)}")

            # Convert to dict for tool input - use explicit conversion to avoid tuple keys
            df_slice_dict = {}
            for col in required_columns:
                if col == "Datetime":
                    # Convert to datetime safely before string formatting.
                    dt_series = self._to_naive_datetime_series(df_slice[col])
                    df_slice_dict[col] = dt_series.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                else:
                    df_slice_dict[col] = df_slice[col].tolist()

            # Debug: Check the resulting dictionary
            print(f"Dictionary keys: {list(df_slice_dict.keys())}")
            print(f"Dictionary key types: {[type(k) for k in df_slice_dict.keys()]}")

            # Format timeframe for display
            display_timeframe = timeframe
            if timeframe.endswith("h"):
                display_timeframe += "our"
            elif timeframe.endswith("m"):
                display_timeframe += "in"
            elif timeframe.endswith("d"):
                display_timeframe += "ay"
            elif timeframe == "1w":
                display_timeframe = "1 week"
            elif timeframe == "1mo":
                display_timeframe = "1 month"

            p_image = static_util.generate_kline_image(df_slice_dict)
            t_image = static_util.generate_trend_image(df_slice_dict)

            # Create initial state
            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": display_timeframe,
                "stock_name": asset_name,
                "pattern_image": p_image["pattern_image"],
                "trend_image": t_image["trend_image"],
            }

            # Run the trading graph
            final_state = self.trading_graph.graph.invoke(initial_state)

            return {
                "success": True,
                "final_state": final_state,
                "asset_name": asset_name,
                "timeframe": display_timeframe,
                "data_length": len(df_slice),
            }

        except Exception as e:
            error_msg = str(e)
            
            # Get current provider from config
            provider = self.config.get("agent_llm_provider", "openai")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            elif provider == "ollama":
                provider_name = "Ollama"
            else:
                provider_name = "Qwen"

            # Check for specific API key authentication errors
            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "success": False,
                    "error": f"❌ Invalid API Key: The {provider_name} API key you provided is invalid or has expired. Please check your API key in the Settings section and try again.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "success": False,
                    "error": f"⚠️ Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"💳 Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"🌐 Network Error: Unable to connect to {provider_name} servers. Please check your internet connection and try again.",
                }
            else:
                return {"success": False, "error": f"❌ Analysis Error: {error_msg}"}

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {"error": results["error"]}

        final_state = results["final_state"]

        # Extract analysis results from state fields
        technical_indicators = final_state.get("indicator_report", "")
        pattern_analysis = final_state.get("pattern_report", "")
        trend_analysis = final_state.get("trend_report", "")
        final_decision_raw = final_state.get("final_trade_decision", "")

        # Extract chart data if available
        pattern_chart = final_state.get("pattern_image", "")
        trend_chart = final_state.get("trend_image", "")
        pattern_image_filename = final_state.get("pattern_image_filename", "")
        trend_image_filename = final_state.get("trend_image_filename", "")

        # Parse final decision
        final_decision = ""
        if final_decision_raw:
            try:
                # Try to extract JSON from the decision
                start = final_decision_raw.find("{")
                end = final_decision_raw.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = final_decision_raw[start:end]
                    decision_data = json.loads(json_str)
                    final_decision = {
                        "decision": decision_data.get("decision", "N/A"),
                        "risk_reward_ratio": decision_data.get(
                            "risk_reward_ratio", "N/A"
                        ),
                        "forecast_horizon": decision_data.get(
                            "forecast_horizon", "N/A"
                        ),
                        "justification": decision_data.get("justification", "N/A"),
                    }
                else:
                    # If no JSON found, return the raw text
                    final_decision = {"raw": final_decision_raw}
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                final_decision = {"raw": final_decision_raw}

        return {
            "success": True,
            "asset_name": results["asset_name"],
            "timeframe": results["timeframe"],
            "data_length": results["data_length"],
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision,
        }

    def get_timeframe_date_limits(self, timeframe: str) -> Dict[str, Any]:
        """Get valid date range limits for a given timeframe."""
        limits = {
            "1m": {"max_days": 7, "description": "1 minute data: max 7 days"},
            "2m": {"max_days": 60, "description": "2 minute data: max 60 days"},
            "5m": {"max_days": 60, "description": "5 minute data: max 60 days"},
            "15m": {"max_days": 60, "description": "15 minute data: max 60 days"},
            "30m": {"max_days": 60, "description": "30 minute data: max 60 days"},
            "60m": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "90m": {"max_days": 60, "description": "90 minute data: max 60 days"},
            "1h": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "4h": {"max_days": 730, "description": "4 hour data: max 730 days"},
            "1d": {"max_days": 730, "description": "1 day data: max 730 days"},
            "5d": {"max_days": 60, "description": "5 day data: max 60 days"},
            "1w": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1wk": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1mo": {"max_days": 730, "description": "1 month data: max 730 days"},
            "3mo": {"max_days": 730, "description": "3 month data: max 730 days"},
        }

        return limits.get(
            timeframe, {"max_days": 730, "description": "Default: max 730 days"}
        )

    def validate_date_range(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
        start_time: str = "00:00",
        end_time: str = "23:59",
    ) -> Dict[str, Any]:
        """Validate date and time range for the given timeframe."""
        try:
            # Create datetime objects with time
            start_datetime_str = f"{start_date} {start_time}"
            end_datetime_str = f"{end_date} {end_time}"

            start = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            end = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")

            if start >= end:
                return {
                    "valid": False,
                    "error": "Start date/time must be before end date/time",
                }

            # Get timeframe limits
            limits = self.get_timeframe_date_limits(timeframe)
            max_days = limits["max_days"]

            # Calculate time difference in days (including fractional days)
            time_diff = end - start
            days_diff = time_diff.total_seconds() / (24 * 3600)  # Convert to days

            if days_diff > max_days:
                return {
                    "valid": False,
                    "error": f"Time range too large. {limits['description']}. Please select a smaller range.",
                    "max_days": max_days,
                    "current_days": round(days_diff, 2),
                }

            return {"valid": True, "days": round(days_diff, 2)}

        except ValueError as e:
            return {"valid": False, "error": f"Invalid date/time format: {str(e)}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current API key by making a simple test call."""
        try:
            # Get provider from config if not provided
            if provider is None:
                provider = self.config.get("agent_llm_provider", "openai")
            
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                
                # Make a simple test call
                _ = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )
                
                provider_name = "OpenAI"
            elif provider == "anthropic":
                from anthropic import Anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("anthropic_api_key", "")
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Anthropic API key is not set. Please update it in the Settings section.",
                    }
                
                client = Anthropic(api_key=api_key)
                
                # Make a simple test call
                _ = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}],
                )
                
                provider_name = "Anthropic"
            elif provider == "ollama":
                base_url = self.config.get("ollama_base_url", "http://localhost:11434/v1")
                api_key = os.environ.get("OLLAMA_API_KEY") or self.config.get(
                    "ollama_api_key", "ollama"
                )
                model = self.config.get("agent_llm_model", "qwen3.5:9b-120k")
                client = OpenAI(api_key=api_key, base_url=base_url)
                _ = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )
                provider_name = "Ollama"
            else:  # qwen
                from langchain_qwq import ChatQwen
                api_key = os.environ.get("DASHSCOPE_API_KEY") or self.config.get("qwen_api_key", "")
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Qwen API key is not set. Please update it in the Settings section.",
                    }
                
                # Make a simple test call using LangChain
                llm = ChatQwen(model="qwen-flash", api_key=api_key)
                _ = llm.invoke([("user", "Hello")])
                
                provider_name = "Qwen"
            return {"valid": True, "message": f"{provider_name} API key is valid"}

        except Exception as e:
            error_msg = str(e)
            
            # Determine provider name for error messages
            if provider is None:
                provider = self.config.get("agent_llm_provider", "openai")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            elif provider == "ollama":
                provider_name = "Ollama"
            else:
                provider_name = "Qwen"

            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "valid": False,
                    "error": f"❌ Invalid API Key: The {provider_name} API key is invalid or has expired. Please update it in the Settings section.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "valid": False,
                    "error": f"⚠️ Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"💳 Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"🌐 Network Error: Unable to connect to {provider_name} servers. Please check your internet connection.",
                }
            else:
                return {"valid": False, "error": f"❌ API Key Error: {error_msg}"}

    def load_custom_assets(self) -> list:
        """Load custom assets from persistent JSON file."""
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            return []
        except Exception as e:
            print(f"Error loading custom assets: {e}")
            return []

    def save_custom_asset(self, symbol: str) -> bool:
        """Save a custom asset symbol persistently (avoid duplicates)."""
        try:
            symbol = symbol.strip()
            if not symbol:
                return False
            if symbol in self.custom_assets:
                return True  # already present
            self.custom_assets.append(symbol)
            # write to file
            with open(self.custom_assets_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_assets, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom asset '{symbol}': {e}")
            return False


# Initialize the analyzer
analyzer = WebTradingAnalyzer()


@app.route("/")
def index():
    """Main landing page - redirect to demo."""
    return render_template("demo_new.html")


@app.route("/demo")
def demo():
    """Demo page with new interface."""
    return render_template("demo_new.html")


@app.route("/output")
def output():
    """Output page with analysis results."""
    # Get results from session or query parameters
    results = request.args.get("results")
    if results:
        try:
            # Handle URL-encoded results
            results = urllib.parse.unquote(results)
            results_data = json.loads(results)
            return render_template("output.html", results=results_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing results: {e}")
            # Fall back to default results

    # Default results if none provided
    default_results = {
        "asset_name": "BTC",
        "timeframe": "1h",
        "data_length": 1247,
        "technical_indicators": "RSI (14): 65.4 - Neutral to bullish momentum\nMACD: Bullish crossover with increasing histogram\nMoving Averages: Price above 50-day and 200-day MA\nBollinger Bands: Price in upper band, showing strength\nVolume: Above average volume supporting price action",
        "pattern_analysis": "Bull Flag Pattern: Consolidation after strong upward move\nGolden Cross: 50-day MA crossing above 200-day MA\nHigher Highs & Higher Lows: Uptrend confirmation\nVolume Pattern: Increasing volume on price advances",
        "trend_analysis": "Primary Trend: Bullish (Long-term)\nSecondary Trend: Bullish (Medium-term)\nShort-term Trend: Consolidating with bullish bias\nADX: 28.5 - Moderate trend strength\nPrice Action: Higher highs and higher lows maintained\nMomentum: Positive divergence on RSI",
        "pattern_chart": "",
        "trend_chart": "",
        "pattern_image_filename": "",
        "trend_image_filename": "",
        "final_decision": {
            "decision": "LONG",
            "risk_reward_ratio": "1:2.5",
            "forecast_horizon": "24-48 hours",
            "justification": "Based on comprehensive analysis of technical indicators, pattern recognition, and trend analysis, the system recommends a LONG position on BTC. The analysis shows strong bullish momentum with key support levels holding, and multiple technical indicators confirming upward movement.",
        },
    }

    return render_template("output.html", results=default_results)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        data_source = data.get("data_source")
        asset = data.get("asset")
        timeframe = data.get("timeframe")
        redirect_to_output = data.get("redirect_to_output", False)

        if data_source != "live":
            return jsonify({"error": "Only live Yahoo Finance data is supported."})

        # Live Yahoo Finance data only
        start_date = data.get("start_date")
        start_time = data.get("start_time", "00:00")
        end_date = data.get("end_date")
        end_time = data.get("end_time", "23:59")
        use_current_time = data.get("use_current_time", False)

        # Create datetime objects for validation
        if start_date:
            start_datetime_str = f"{start_date} {start_time}"
            try:
                start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
                return jsonify({"error": "Invalid start date/time format."})

            if start_dt > datetime.now():
                return jsonify({"error": "Start date/time cannot be in the future."})

        if end_date:
            if use_current_time:
                end_dt = datetime.now()
            else:
                end_datetime_str = f"{end_date} {end_time}"
                try:
                    end_dt = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    return jsonify({"error": "Invalid end date/time format."})

                if end_dt > datetime.now():
                    return jsonify({"error": "End date/time cannot be in the future."})

            if start_date and start_dt and end_dt and end_dt < start_dt:
                return jsonify(
                    {"error": "End date/time cannot be earlier than start date/time."}
                )

        # Fetch data with datetime objects
        df = analyzer.fetch_yfinance_data_with_datetime(
            asset, timeframe, start_dt, end_dt
        )
        if df.empty:
            return jsonify({"error": "No data available for the specified parameters"})

        if os.environ.get("QUANTAGENT_SMOKE_ANALYZE", "").lower() in ("1", "true", "yes"):
            return jsonify(
                {
                    "success": True,
                    "smoke": True,
                    "rows": int(len(df)),
                    "asset": asset,
                    "timeframe": timeframe,
                }
            )

        display_name = analyzer.asset_mapping.get(asset, asset)
        if display_name is None:
            display_name = asset
        results = analyzer.run_analysis(df, display_name, timeframe)
        formatted_results = analyzer.extract_analysis_results(results)

        # If redirect is requested, return redirect URL with results
        if redirect_to_output:
            if formatted_results.get("success", False):
                # Create a version without base64 images for URL encoding
                # Base64 images are too large for URL parameters
                url_safe_results = formatted_results.copy()
                url_safe_results["pattern_chart"] = ""  # Remove base64 data
                url_safe_results["trend_chart"] = ""  # Remove base64 data

                # Encode results for URL
                results_json = json.dumps(url_safe_results)
                encoded_results = urllib.parse.quote(results_json)
                redirect_url = f"/output?results={encoded_results}"

                # Store full results (with images) in session or temporary storage
                # For now, we'll pass them back in the response for the frontend to handle
                return jsonify(
                    {
                        "redirect": redirect_url,
                        "full_results": formatted_results,  # Include images in response body
                    }
                )
            else:
                return jsonify(
                    {"error": formatted_results.get("error", "Analysis failed")}
                )

        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/files/<asset>/<timeframe>")
def get_files(asset, timeframe):
    """API endpoint to get available files for an asset/timeframe."""
    try:
        files = analyzer.get_available_files(asset, timeframe)
        file_list = []

        for i, file_path in enumerate(files):
            match = re.search(r"_(\d+)\.csv$", file_path.name)
            file_number = match.group(1) if match else "N/A"
            file_list.append(
                {"index": i, "number": file_number, "name": file_path.name}
            )

        return jsonify({"files": file_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/save-custom-asset", methods=["POST"])
def save_custom_asset():
    """Save a custom asset symbol server-side for persistence."""
    try:
        data = request.get_json()
        symbol = (data.get("symbol") or "").strip()
        if not symbol:
            return jsonify({"success": False, "error": "Symbol required"}), 400

        ok = analyzer.save_custom_asset(symbol)
        if not ok:
            return jsonify({"success": False, "error": "Failed to save symbol"}), 500

        return jsonify({"success": True, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/custom-assets", methods=["GET"])
def custom_assets():
    """Return server-persisted custom assets."""
    try:
        return jsonify({"custom_assets": analyzer.custom_assets or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets")
def get_assets():
    """API endpoint to get available assets."""
    try:
        assets = analyzer.get_available_assets()
        asset_list = []

        for asset in assets:
            asset_list.append(
                {"code": asset, "name": analyzer.asset_mapping.get(asset, asset)}
            )

        # Include server-persisted custom assets at the end
        for custom in analyzer.custom_assets:
            asset_list.append({"code": custom, "name": custom})

        return jsonify({"assets": asset_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/timeframe-limits/<timeframe>")
def get_timeframe_limits(timeframe):
    """API endpoint to get date range limits for a timeframe."""
    try:
        limits = analyzer.get_timeframe_date_limits(timeframe)
        return jsonify(limits)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-date-range", methods=["POST"])
def validate_date_range():
    """API endpoint to validate date and time range for a timeframe."""
    try:
        data = request.get_json()
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        timeframe = data.get("timeframe")
        start_time = data.get("start_time", "00:00")
        end_time = data.get("end_time", "23:59")

        if not all([start_date, end_date, timeframe]):
            return jsonify({"error": "Missing required parameters"})

        validation = analyzer.validate_date_range(
            start_date, end_date, timeframe, start_time, end_time
        )
        return jsonify(validation)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/update-provider", methods=["POST"])
def update_provider():
    """API endpoint to update LLM provider."""
    try:
        data = request.get_json()
        provider = data.get("provider", "openai")

        if provider not in ["openai", "anthropic", "qwen", "ollama"]:
            return jsonify(
                {"error": "Provider must be 'openai', 'anthropic', 'qwen', or 'ollama'"}
            )

        print(f"Updating provider to: {provider}")

        # Update config in both analyzer and trading_graph
        analyzer.config["agent_llm_provider"] = provider
        analyzer.config["graph_llm_provider"] = provider
        analyzer.trading_graph.config["agent_llm_provider"] = provider
        analyzer.trading_graph.config["graph_llm_provider"] = provider
        
        # Update model names if switching providers
        if provider == "anthropic":
            # Set default Claude models if not already set to Anthropic models
            if not analyzer.config["agent_llm_model"].startswith("claude"):
                analyzer.config["agent_llm_model"] = "claude-haiku-4-5-20251001"
            if not analyzer.config["graph_llm_model"].startswith("claude"):
                analyzer.config["graph_llm_model"] = "claude-haiku-4-5-20251001"
        elif provider == "qwen":
            # Ollama Qwen tags contain ':'; cloud DashScope models do not
            am = str(analyzer.config["agent_llm_model"])
            gm = str(analyzer.config["graph_llm_model"])
            if (not am.startswith("qwen")) or (":" in am):
                analyzer.config["agent_llm_model"] = "qwen3-max"
            if (not gm.startswith("qwen")) or (":" in gm):
                analyzer.config["graph_llm_model"] = "qwen3-vl-plus"
        elif provider == "ollama":
            analyzer.config["agent_llm_model"] = "qwen3.5:9b-120k"
            analyzer.config["graph_llm_model"] = "qwen3.5:9b-120k"
        else:
            # Set default OpenAI models if not already set to OpenAI models
            am = str(analyzer.config["agent_llm_model"])
            gm = str(analyzer.config["graph_llm_model"])
            if am.startswith(("claude", "qwen")) or ":" in am:
                analyzer.config["agent_llm_model"] = "gpt-4o-mini"
            if gm.startswith(("claude", "qwen")) or ":" in gm:
                analyzer.config["graph_llm_model"] = "gpt-4o"
        
        analyzer.trading_graph.config.update(analyzer.config)

        # Refresh the trading graph with new provider
        analyzer.trading_graph.refresh_llms()

        print(f"Provider updated to {provider} successfully")
        print(f"graph_llm_model updated to {analyzer.config['graph_llm_model']} successfully")
        print(f"agent_llm updated to {analyzer.config['agent_llm_model']} successfully")
        return jsonify({"success": True, "message": f"Provider updated to {provider}"})

    except Exception as e:
        print(f"Error in update_provider: {str(e)}")
        return jsonify({"error": str(e)})


@app.route("/api/update-api-key", methods=["POST"])
def update_api_key():
    """API endpoint to update API key for OpenAI or Anthropic."""
    try:
        data = request.get_json()
        new_api_key = data.get("api_key")
        provider = data.get("provider", "openai")  # Default to "openai" for backward compatibility

        if not new_api_key:
            return jsonify({"error": "API key is required"})

        if provider not in ["openai", "anthropic", "qwen", "ollama"]:
            return jsonify(
                {"error": "Provider must be 'openai', 'anthropic', 'qwen', or 'ollama'"}
            )

        print(f"Updating {provider} API key to: {new_api_key[:8]}...{new_api_key[-4:]}")

        # Update the environment variable
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = new_api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = new_api_key
        elif provider == "qwen":
            os.environ["DASHSCOPE_API_KEY"] = new_api_key
        elif provider == "ollama":
            os.environ["OLLAMA_API_KEY"] = new_api_key

        # Update the API key in the trading graph
        analyzer.trading_graph.update_api_key(new_api_key, provider=provider)

        print(f"{provider} API key updated successfully")
        return jsonify({"success": True, "message": f"{provider.capitalize()} API key updated successfully"})

    except Exception as e:
        print(f"Error in update_api_key: {str(e)}")
        return jsonify({"error": str(e)})


@app.route("/api/get-api-key-status")
def get_api_key_status():
    """API endpoint to check if API key is set for a provider."""
    try:
        provider = request.args.get("provider", "openai")
        
        # First check environment variables
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            # Fallback to config if not in environment
            if not api_key and hasattr(analyzer, 'config'):
                api_key = analyzer.config.get("api_key", "")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            # Fallback to config if not in environment
            if not api_key and hasattr(analyzer, 'config'):
                api_key = analyzer.config.get("anthropic_api_key", "")
        elif provider == "qwen":
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            # Fallback to config if not in environment
            if not api_key and hasattr(analyzer, 'config'):
                api_key = analyzer.config.get("qwen_api_key", "")
        elif provider == "ollama":
            api_key = os.environ.get("OLLAMA_API_KEY", "")
            if not api_key and hasattr(analyzer, "config"):
                api_key = analyzer.config.get("ollama_api_key", "ollama")
            if not api_key:
                api_key = "ollama"
            masked_key = (
                api_key[:3] + "..." + api_key[-3:] if len(api_key) > 12 else "***"
            )
            return jsonify({"has_key": True, "masked_key": masked_key})
        else:
            api_key = ""
        
        if api_key and api_key != "your-openai-api-key-here" and api_key != "":
            # Return masked version for security
            masked_key = (
                api_key[:3] + "..." + api_key[-3:] if len(api_key) > 12 else "***"
            )
            return jsonify({"has_key": True, "masked_key": masked_key})
        else:
            return jsonify({"has_key": False})
    except Exception as e:
        print(f"Error in get_api_key_status: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "has_key": False})


@app.route("/api/images/<image_type>")
def get_image(image_type):
    """API endpoint to serve generated images."""
    try:
        if image_type == "pattern":
            image_path = "kline_chart.png"
        elif image_type == "trend":
            image_path = "trend_graph.png"
        elif image_type == "pattern_chart":
            image_path = "pattern_chart.png"
        elif image_type == "trend_chart":
            image_path = "trend_chart.png"
        else:
            return jsonify({"error": "Invalid image type"})

        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"})

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-api-key", methods=["POST"])
def validate_api_key():
    """API endpoint to validate the current API key."""
    try:
        data = request.get_json() or {}
        provider = data.get("provider") or analyzer.config.get("agent_llm_provider", "openai")
        validation = analyzer.validate_api_key(provider=provider)
        return jsonify(validation)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve static assets from the assets folder."""
    try:
        return send_file(f"assets/{filename}")
    except FileNotFoundError:
        return jsonify({"error": "Asset not found"}), 404


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # Create static directory if it doesn't exist
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    port = int(os.environ.get("QUANTAGENT_PORT", os.environ.get("PORT", "5000")))
    app.run(debug=True, host="127.0.0.1", port=port)
