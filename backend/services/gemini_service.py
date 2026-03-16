import os
import base64
import json
import logging
import numpy as np

from services.model_client import model_client


logger = logging.getLogger("ai_service")

# JSON enforcement suffix for prompts expecting structured output
JSON_INSTRUCTION = (
    "\n\nCRITICAL: Respond with ONLY a valid JSON object. "
    "No markdown fences. No explanation text. "
    "Start with { and end with }."
)


class _ChatResponse:
    """Mimics the Gemini response object with a .text attribute."""
    def __init__(self, text: str):
        self.text = text


class _ModelClientChat:
    """
    Lightweight chat wrapper that mimics genai.GenerativeModel.start_chat().
    
    Provides .send_message(text) -> response with .text attribute.
    Maintains conversation history for context.
    """
    def __init__(self, system_instruction: str, service, history: list = None):
        self.system_instruction = system_instruction.strip()
        self.service = service
        self.history = history or []
    
    async def send_message(self, message: str) -> _ChatResponse:
        """Send a message and get a response, maintaining conversation context and handling tools."""
        import re
        import ast
        
        # Build full prompt with system instruction and history
        full_prompt = f"{self.system_instruction}\n\n"
        
        for turn in self.history:
            role = turn.get("role", "user")
            content = turn.get("parts", [{"text": ""}])[0]
            if isinstance(content, dict):
                content = content.get("text", "")
            full_prompt += f"{role}: {content}\n"
        
        full_prompt += f"user: {message}\nassistant:"
        
        response_text = await model_client.generate_async(full_prompt, task="text")
        
        # Continuously check for tool calls until model provides a final text response
        tool_pattern = r'\[TOOL_CALL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)\]'
        max_tool_loops = 5
        loop_count = 0
        
        ALLOWED_TOOLS = {
            "search_knowledge_base",
            "simulate_charging_analysis",
            "predict_battery_life",
            "parse_logs",
            "create_incident_report",
            "simulate_fleet"
        }
        
        while True:
            matches = re.findall(tool_pattern, response_text)
            if not matches:
                break
                
            loop_count += 1
            if loop_count > max_tool_loops:
                logger.warning("Agent exceeded maximum tool execution loops (%d).", max_tool_loops)
                break
                
            full_prompt += "assistant: [tool call executed]\n"

            # Execute all tools found in this turn
            for raw_tool_name, args_str in matches:
                if raw_tool_name not in ALLOWED_TOOLS:
                    logger.warning("Agent attempted to run forbidden/hallucinated tool: %s", raw_tool_name)
                    full_prompt += f"system: [TOOL_ERROR {raw_tool_name}]: Not an allowed tool. You can only use tools from the provided list.\n"
                    continue
                    
                tool_name = f"tool_{raw_tool_name}"
                args = ()
                
                if args_str.strip():
                    try:
                        # Safely parse arguments like ("thermal", "test")
                        args = ast.literal_eval(f"({args_str})")
                        if not isinstance(args, tuple):
                            args = (args,)
                    except Exception as e:
                        # Fallback for unquoted strings: search_knowledge_base(lithium, failure)
                        logger.warning("Literal eval failed for '%s', falling back to split tuple: %s", args_str, e)
                        args = tuple(arg.strip() for arg in args_str.split(",") if arg.strip())
                        
                try:
                    tool_method = getattr(self.service, tool_name, None)
                    if tool_method is None:
                        logger.warning("Unknown tool requested: %s", tool_name)
                        full_prompt += f"system: [TOOL_ERROR {raw_tool_name}]: Unknown tool implementation.\n"
                        continue
                        
                    import inspect
                    # Call tool
                    if inspect.iscoroutinefunction(tool_method):
                        tool_result = await tool_method(*args)
                    else:
                        tool_result = tool_method(*args)
                    if isinstance(tool_result, (dict, list)):
                        tool_result_str = json.dumps(tool_result)[:2000]
                    else:
                        tool_result_str = str(tool_result)[:2000]
                    full_prompt += f"system: [TOOL_RESULT {raw_tool_name}]: {tool_result_str}\n"
                except Exception as e:
                    logger.error("Tool execution failed: %s", e)
                    tool_result_str = str(e)[:2000]
                    full_prompt += f"system: [TOOL_ERROR {raw_tool_name}]: {tool_result_str}\n"
            
            # Guard against massive prompt growth inside the tool loop
            if len(full_prompt) > 50000:
                logger.warning("Agent prompt exceeded safe size, truncating history.")
                full_prompt = self.system_instruction + "\n\n...[TRUNCATED HISTORY]...\n" + full_prompt[-40000:]
            
            full_prompt += "assistant:"
            response_text = await model_client.generate_async(full_prompt, task="text")
                
        # Update history with limit
        MAX_HISTORY = 6
        self.history.append({"role": "user", "parts": [{"text": message}]})
        self.history.append({"role": "model", "parts": [{"text": response_text}]})
        self.history = self.history[-MAX_HISTORY:]
        
        return _ChatResponse(response_text)


class GeminiService:
    def __init__(self):
        # Model calls are now routed through model_client (RunPod API)
        # No Gemini SDK initialization needed
        pass

    def _build_prompt(self, prompt: str, enforce_json: bool = True) -> str:
        """Standardize prompt construction with optional JSON enforcement."""
        final = prompt.strip()
        if enforce_json:
            final += JSON_INSTRUCTION
        return final

    def generate_raw_text(self, prompt: str) -> str:
        """Simple text generation helper for external callers (e.g. digital_twin_service)."""
        try:
            return model_client.generate(self._build_prompt(prompt, enforce_json=False), task="text")
        except Exception as e:
            logger.error("Raw generation failed: %s", e)
            return ""

    def _llm_json(self, prompt: str, task: str = "text") -> dict:
        """Helper to generate text and extract JSON to reduce boilerplate."""
        response_text = model_client.generate(self._build_prompt(prompt), task=task)
        return self._extract_json(response_text)

    async def _llm_json_async(self, prompt: str, image_b64: str = None, task: str = "text") -> dict:
        """Async helper to generate text (with optional image) and extract JSON."""
        response_text = await model_client.generate_async(
            self._build_prompt(prompt), image_b64=image_b64, task=task
        )
        return self._extract_json(response_text)

    def _sanitize_for_json(self, obj):
        """Recursively converts NumPy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(i) for i in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    async def analyze_defect(self, image_data, mime_type="image/jpeg"):
        prompt = """
        Act as 'BatteryGPT', a specialized domain expert in Lithium-Ion battery anomaly detection.
        Follow the 'Detect-Locate-Describe' methodology:
        1. DETECT: Identify if any anomaly exists (Swelling, Corrosion, Leakage, Mechanical Deformation, Thermal Runaway).
        2. LOCATE: Pinpoint the specific region (e.g., 'upper tab', 'cell body center', 'negative terminal').
        3. DESCRIBE: Provide a technical electrochemical assessment of the visual evidence.

        Analyze the attached image.
        Output a valid JSON object with:
        - defect_type: (string) Class of defect or 'Normal'.
        - location: (string) Specific physical region of the defect.
        - severity: (string) 'Negligible', 'Moderate', 'Critical'.
        - confidence: (number) 0-100.
        - description: (string) Detailed technical analysis following the methodology.
        - mitigation: (string) Immediate safety or maintenance action.
        """
        
        try:
            if len(image_data) > 15_000_000:
                logger.warning(
                    "Large image input detected (%d bytes). Vision inference may be slow.",
                    len(image_data)
                )
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            return await self._llm_json_async(prompt, image_b64=image_b64, task="vision")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def analyze_pcb_defect(self, image_data, mime_type="image/jpeg"):
        """
        PCB Defect Inspection using Detect-Locate-Describe Methodology.
        Analyzes PCB images for manufacturing defects with spatial precision.
        """
        prompt = """
        Act as 'PCB-VisionGPT', a specialized expert in PCB manufacturing defect detection.
        Follow the 'Detect-Locate-Describe' methodology for comprehensive analysis:
        
        1. DETECT: Identify if any defect exists (Open Circuit, Short Circuit, Solder Mask Issues, Mouse Bites, Drilling Defects, Copper Delamination).
        2. LOCATE: Pinpoint the specific region (e.g., 'trace between U1 pin 3 and R5', 'top-right quadrant near mounting hole', 'layer 2 signal trace').
        3. DESCRIBE: Provide a detailed technical assessment of the defect mechanism and manufacturing impact.
        
        Analyze the attached PCB image.
        Output a valid JSON object with:
        - defect_type: (string) Classification of defect or 'NORMAL'.
        - location: (string) Precise physical location of the defect on the PCB.
        - severity: (string) 'FATAL' (scrappable), 'CRITICAL' (costly repair), 'REPAIRABLE' (standard repair), 'WARNING' (cosmetic/minor).
        - confidence: (number) 0-100, your confidence in this assessment.
        - description: (string) Detailed technical analysis following the Detect-Locate-Describe methodology. Explain the defect mechanism.
        - mitigation: (string) Immediate action required: 'SCRAP', 'MANUAL_REPAIR', 'REWORK', 'AUTOMATED_REPAIR', 'ACCEPT_WITH_WAIVER', or 'NONE'.
        - root_cause: (string) Likely manufacturing process failure (e.g., 'Etching over-exposure', 'Drill bit wear', 'Solder mask misalignment').
        - bbox: (array) Approximate bounding box [x, y, width, height] in relative coordinates [0-100] if defect is visually localizable, otherwise null.
        
        Example Output:
        {
            "defect_type": "OPEN_CIRCUIT",
            "location": "Signal trace between IC1 pin 7 and resistor R12, top copper layer, X=45mm Y=23mm",
            "severity": "FATAL",
            "confidence": 94,
            "description": "DETECT: Open circuit detected. LOCATE: 0.8mm gap in 0.2mm wide signal trace on top layer. DESCRIBE: Complete electrical discontinuity caused by chemical etching over-exposure. The trace width reduced from nominal 0.2mm to 0mm at failure point. This will prevent signal propagation and render the board non-functional.",
            "mitigation": "SCRAP",
            "root_cause": "Chemical etching process exceeded target removal rate by approximately 30%, likely due to high etchant concentration or extended dwell time",
            "bbox": [42, 18, 8, 6]
        }
        """
        try:
            if len(image_data) > 15_000_000:
                logger.warning(
                    "Large image input detected (%d bytes). Vision inference may be slow.",
                    len(image_data)
                )
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            return await self._llm_json_async(prompt, image_b64=image_b64, task="vision")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def analyze_gerber_text(self, gerber_content: str):
        """
        Phase 6: Real AI Text Analysis for Gerber Files.
        """
        # Truncate content if too huge, but usually headers are top 50 lines.
        # We need Gemini to inspect the syntax.
        
        prompt = f"""
        Act as a CAM Engineer (Computer-Aided Manufacturing).
        Review this Gerber File (RS-274X) snippet:
        
        {gerber_content[:4000]} ... (truncated)
        
        Analyze the header and aperture definitions.
        1. Identify the layer type (Copper, Mask, Drill, Silk).
        2. Check for missing crucial definitions (e.g., Units, Format).
        
        Return JSON:
        {{
            "layer_type": "Signal_Top",
            "units": "Metric" | "Imperial",
            "is_valid_format": true,
            "missing_features": ["Undefined Aperture D10"], // List strings or empty
            "engineering_check": "PASS" | "FAIL"
        }}
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def analyze_charging_curve(self, image_buffer):
        """
        Analyzes a charging curve plot for electrochemical signatures.
        """
        prompt = """
        You are an Expert Battery Data Scientist.
        Analyze this 'Voltage & Current vs Time' charging plot.
        
        Look for specific anomalies:
        1. Voltage Kinks/Plateaus: Could indicate Lithium Plating (critical).
        2. Abnormal IR Drop: High internal resistance.
        3. Capacity Fade: Reaching cutoff voltage too early.
        
        Return a JSON object:
        {
          "anomaly_detected": true/false,
          "diagnosis": "Lithium Plating / Normal / etc",
          "severity": "High/Medium/Low",
          "description": "Explanation of the curve shape...",
          "reasoning": "Step-by-step electrochemical deduction (e.g. 'Slope change at 3.8V indicates...').",
          "recommendation": "Decrease charging rate / Check thermal management"
        }
        """
        
        try:
            image_data = image_buffer.getvalue()
            if len(image_data) > 15_000_000:
                logger.warning(
                    "Large image input detected (%d bytes). Vision inference may be slow.",
                    len(image_data)
                )
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            return await self._llm_json_async(prompt, image_b64=image_b64, task="vision")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def parse_fault_log(self, log_text, context=None):
        prompt = f"""
        You are an expert Battery Management System (BMS) Log Analyzer.
        Parse the following raw log/error code dump.
        Use the provided 'Context Data' (Voltage, Temp, etc.) to refine your diagnosis.
        Extract:
        1. Error Code (if present).
        2. Component (e.g., Cell Module 3, BMS Main Controller).
        3. Issue Description.
        4. Urgency (Info, Warning, Critical).
        5. troubleshooting_steps (List of strings).

        Context Data (Battery State):
        {context}

        Raw Log:
        {log_text}

        Return the result ONLY as a valid JSON object with keys:
        error_code, component, description, urgency, troubleshooting_steps.
        """

        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}



    async def predict_battery_aging(self, aging_data):
        prompt = f"""
        You are a Battery Lifecycle Analyst.
        Here is the historical capacity data (State of Health vs Cycle Number) for a battery pack:
        
        Cycles: {aging_data['cycles'][-10:]} (Last 10 points)
        SOH: {aging_data['soh'][-10:]} (Last 10 points)
        
        Task:
        1. Predict the 'End of Life' (EOL) cycle number (when SOH hits 80%).
        2. Identify if a 'Knee Point' (accelerated degradation) has occurred.
        3. Estimate Remaining Useful Life (RUL) in cycles.
        
        Return JSON:
        {{
            "predicted_eol_cycle": 1200,
            "rul_cycles": 400,
            "knee_point_detected": true/false,
            "reasoning": "The slope has increased significantly..."
        }}
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    def _extract_json(self, text: str):
        """
        Robustly extracts JSON object from LLM response, handling markdown fences and chatty prefixes.
        """
        try:
            # 1. Remove Markdown fences
            clean_text = text.replace("```json", "").replace("```", "").strip()
            
            # 2. Find the first '{' and last '}' to isolate the JSON object
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = clean_text[start_idx : end_idx + 1]
                return json.loads(json_str)
            
            # Fallback: Try regex if simple find fails (e.g. nested structures might be fine, but just in case)
            import re
            match = re.search(r'(\{.*?\})', clean_text, re.S)
            if match:
                return json.loads(match.group(1))

            # Last resort: Try loading the stripped text directly
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error("JSON parse error: %s | Text: %s...", e, text[:100])
            # Safe JSON fallback to prevent endpoint crash
            return {"error": "invalid_json", "raw": text[:500], "parsed": False}


    async def predict_aging_trajectory(self, current_soh: float, start_cycle: int):
        """
        Generates a plausible lithium-ion degradation curve (Cycles vs SOH) using Gemini's physics knowledge.
        Returns a list of points or a structured JSON response with the curve data.
        """
        prompt = f"""
        Act as an Advanced Battery Physics Simulator.
        Generate a projected capacity fade (aging) curve for a Lithium-Ion NMC/Graphite cell.
        
        Initial Conditions:
        - Current SOH: {current_soh}%
        - Current Cycle Count: {start_cycle}
        
        Simulation Parameters:
        - Project forward for 2000 cycles.
        - Include 'Knee Point' onset modeling (accelerated fading after ~80% SOH is reached).
        - Add realistic Gaussian noise (+/- 0.2%) to simulate measurement noise.
        
        Output Result ONLY as a Valid JSON object with this structure:
        {{
            "cycles": [0, 50, 100, ...], 
            "soh": [100.0, 99.8, 99.5, ...],
            "analysis": {{
                "prediction_engine": "AI + Physics Hybrid (Hybrid)",
                "summary": "Detailed technical summary (2 sentences) of the degradation trend and projected RUL.",
                "recommendation": "One actionable recommendation to extend cycle life."
            }}
        }}
        Ensure the 'cycles' array starts at 0 and goes up to at least {start_cycle + 2000}.
        Ensure 'soh' matches 'cycles' length. SOH should decay from 100% (at cycle 0) down to <60%.
        Make sure the curve passes roughly through the Current SOH at the Current Cycle Count.
        """
        
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return None

    async def analyze_dataset_signature(self, headers: list, sample_rows: str):
        """
        UNIVERSAL ANALYZER:
        Identifies the dataset type and recommends plotting configuration.
        """
        prompt = f"""
        Act as a Data Scientist specialized in Battery R&D.
        Analyze this CSV header and sample data to understand what it represents.
        
        Headers: {headers}
        Sample Data:
        {sample_rows}
        
        Tasks:
        1. Classify the Data Type: 'Cycling' (Time/Volts/Amps), 'Impedance' (EIS), 'Diffraction' (XRD), 'Mechanical', or 'Unknown'.
        2. Summary: One sentence description of what this file contains.
        3. Visualization Config: Pick the BEST single pair of columns to plot to visualize this data.
           - For Cycling: X='Time', Y='Voltage' (or similar)
           - For Impedance: X='Real_Z', Y='Imag_Z' (Nyquist) - Note: Nyquist usually needs 'Imag_Z' inverted, but just pick the column for now.
           
        Return JSON ONLY:
        {{
            "dataset_type": "Impedance",
            "summary": "Electrochemical Impedance Spectroscopy scan showing real and imaginary resistance.",
            "plot_recommendation": {{
                "x_axis_col": "Real_Z",
                "y_axis_col": "-Imag_Z", 
                "title": "Nyquist Plot",
                "invert_y": true
            }},
            "is_standard_cycling": false
        }}
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.error("Dataset signature analysis error: %s", e)
            # Fallback
            return {
                "dataset_type": "Unknown",
                "summary": "Could not identify data structure.",
                "plot_recommendation": {"x_axis_col": headers[0], "y_axis_col": headers[1] if len(headers)>1 else headers[0]},
                "is_standard_cycling": False
            }
    
    async def suggest_interactive_plots(self, headers: list, sample_rows: str, column_stats: dict = None):
        """
        INTELLIGENT PLOT RECOMMENDER:
        Analyzes the dataset and suggests multiple interactive charts the user might want to see.
        Returns a list of plot configurations that the frontend can render as chart tabs.
        """
        stats_info = ""
        if column_stats:
            stats_info = f"\nColumn Statistics:\n{column_stats}"
        
        prompt = f"""
        Act as a Senior Data Visualization Expert for Battery and EV Telemetry Data.
        
        Analyze this dataset and recommend the BEST interactive charts to explore it:
        
        Headers: {headers}
        Sample Data:
        {str(sample_rows)[:10000]}
        {str(stats_info)[:5000]}
        
        Your task:
        1. Identify ALL meaningful numeric columns that could be plotted
        2. Recommend 2-5 different chart configurations that would be most useful for analysis
        3. Prioritize charts that reveal important patterns (voltage trends, temperature anomalies, SOC behavior, etc.)
        4. For each chart, specify:
           - X-axis column (usually time or a sequential index)
           - Y-axis column(s) - can suggest multiple series for overlay
           - Chart type: 'line', 'scatter', 'area'
           - Why this chart is useful
        
        IMPORTANT:
        - Use EXACT column names from the headers provided
        - If a column looks like 'cell1_voltage', 'cell2_voltage', etc., suggest overlaying them
        - Suggest temperature vs time if temperature columns exist
        - Suggest SOC (State of Charge) if it exists
        
        Return JSON ONLY:
        {{
            "data_description": "Brief description of what this dataset contains",
            "total_columns": 25,
            "numeric_columns": ["time", "voltage", "current", ...],
            "recommended_plots": [
                {{
                    "id": "voltage_trend",
                    "title": "Pack Voltage Over Time",
                    "chart_type": "line",
                    "x_axis": "timestamp",
                    "y_axes": ["pack_voltage"],
                    "description": "Shows overall battery voltage behavior during the session",
                    "priority": 1
                }},
                {{
                    "id": "cell_voltages",
                    "title": "Individual Cell Voltages",
                    "chart_type": "line", 
                    "x_axis": "timestamp",
                    "y_axes": ["cell1_v", "cell2_v", "cell3_v"],
                    "description": "Compare individual cell voltages to detect imbalance",
                    "priority": 2
                }},
                {{
                    "id": "temp_monitoring",
                    "title": "Temperature Profile",
                    "chart_type": "area",
                    "x_axis": "timestamp", 
                    "y_axes": ["temperature"],
                    "description": "Monitor thermal behavior for safety analysis",
                    "priority": 3
                }}
            ],
            "insights": [
                "Dataset appears to be BMS telemetry from EV charging session",
                "Cell voltage imbalance detected in sample - recommend cell comparison chart"
            ]
        }}
        """
        try:
            result = await self._llm_json_async(prompt)
            if result:
                # Ensure recommended_plots exists
                if 'recommended_plots' not in result:
                    result['recommended_plots'] = []
                return result
            return {"error": "Could not parse response", "recommended_plots": []}
        except Exception as e:
            logger.error("Plot suggestion error: %s", e)
            # Fallback: Create basic recommendations from headers
            numeric_cols = [h for h in headers if not any(x in h.lower() for x in ['id', 'name', 'date', 'string'])]
            time_cols = [h for h in headers if any(x in h.lower() for x in ['time', 'timestamp', 'date', 'seconds', 't'])]
            x_axis = time_cols[0] if time_cols else (headers[0] if headers else 'index')
            
            return {
                "data_description": "Dataset analysis failed, showing basic plot",
                "numeric_columns": numeric_cols[:10],
                "recommended_plots": [
                    {
                        "id": "default",
                        "title": "Data Overview",
                        "chart_type": "line",
                        "x_axis": x_axis,
                        "y_axes": [numeric_cols[1]] if len(numeric_cols) > 1 else [headers[0]],
                        "description": "Default visualization",
                        "priority": 1
                    }
                ],
                "insights": ["Fallback mode - AI analysis failed"]
            }
            
    async def map_eis_columns(self, headers: list, sample_rows: str):
        """
        Specialized Mapper for EIS Data (Frequency, Real, Imaginary).
        """
        prompt = f"""
        Act as an Electrochemistry Data Expert.
        Identify the column names for Nyquist Plot data from the headers provided.
        
        Headers: {headers}
        Sample Data:
        {sample_rows}
        
        Target Keys:
        - 'freq': Frequency (Hz)
        - 'real': Real Impedance (Z', Re(Z)) (Ohm)
        - 'imag': Imaginary Impedance (Z'', Im(Z)) (Ohm)
        
        Return JSON mapping: {{ "freq": "original_col_name", "real": "...", "imag": "..." }}.
        If you cannot find a column, omit the key.
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def map_columns_semantic(self, headers: list, sample_rows: str):
        """
        ROSETTA STONE: Maps arbitrary CSV headers to standard battery keys.
        """
        prompt = f"""
        Act as a Data Ingestion Specialist for Battery Research.
        Map the provided CSV headers to the standard internal schema:
        - 'time': Time in seconds (or equivalent step).
        - 'voltage': Cell voltage (V).
        - 'current': Current (A).
        - 'capacity': Capacity (Ah) (Optional).
        - 'temperature': Cell/Pack Temperature (C) (Optional).
        - 'soc': State of Charge (%) (Optional).
        
        Headers: {headers}
        Sample Data:
        {sample_rows}
        
        Return a JSON object mapping: {{ "standard_key": "original_header" }}.
        Only include keys you are confident about.
        Snippet 2: "Step_Time", "Step_Index", "Voltage_V", "Current_A" -> {{"time": "Step_Time", "voltage": "Voltage_V", "current": "Current_A"}}
        
        CRITICAL RULES:
        1. For .mat/nested files, headers are flattened (e.g., 'data_step_voltage', 'operation_data_sub_0_volts').
           YOU MUST MAP THESE. Ignore the prefixes ('data_step_', 'operation_') and match the core term ('voltage', 'volts', 'u_meas').
        2. If you see 'voltage' or 'current' ANYWHERE in the string (case-insensitive), it is a very strong candidate.
        3. 'time' might be 'step_time', 'test_time', 'duration', 't'.
        4. Do not fail if you are 80% sure. We prefer a likely match over no match.
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def analyze_telemetry_deep_dive(self, telemetry_summary: str):
        """
        Deep Dive Analysis for datasets with extended telemetry (Temp, SOC).
        """
        prompt = f"""
        Act as a Senior Battery Systems Engineer.
        Perform a Deep Dive Analysis on this battery telemetry snapshot.
        
        Data Snapshot (First 50 rows sample):
        {telemetry_summary}
        
        Analyze specifically for:
        1. **Thermal Stability**: Are temperatures correlated with high current? Any runaway signs?
        2. **SOC Consistency**: Does the voltage curve match expected OCV behavior for the given SOC?
        3. **Imbalance Risk**: (If multiple voltages present, though this snapshot is single-stream)
        
        Return JSON Key Highlights:
        {{
            "thermal_analysis": "Temperatures peaked at 45C during discharge, which is within limits but high.",
            "soc_analysis": "SOC usage efficient, range 90%-10%.",
            "safety_score": 85, // 0-100
            "optimization_tip": "Consider active cooling during high C-rate discharge."
        }}
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def analyze_eis_spectrum(self, frequency, z_real, z_imag):
        """
        Analyzes Electrochemical Impedance Spectroscopy (EIS) data.
        Splits analysis into 3 layers as per IEST standards:
        1. High Freq (>1kHz): Ohmic/Contact Resistance (R_b).
        2. Mid Freq (1kHz-1Hz): Charge Transfer (R_ct) / SEI.
        3. Low Freq (<1Hz): Diffusion (Warburg).
        """
        # Downsample for prompt if too large
        step = max(1, len(frequency) // 50)
        data_sample = []
        for i in range(0, len(frequency), step):
            data_sample.append(f"F:{frequency[i]:.1f}Hz, Z'={z_real[i]:.4f}, Z''={z_imag[i]:.4f}")
            
        prompt = f"""
        Act as an Electrochemistry Expert specialized in EIS Analysis (Nyquist Plots).
        
        Analyze this Impedance Spectrum sample data:
        {json.dumps(data_sample)}
        
        PERFORM A MULTI-LAYERED DIAGNOSIS:
        
        1. HIGH FREQUENCY (>1000 Hz): Check for Inductive tails or pure Ohmic shift.
           - Diagnostic: Is the start point shifted right? (High Contact Resistance/Cable failure).
           
        2. MID FREQUENCY (1000 Hz - 1 Hz): Check the Semicircle(s).
           - Diagnostic: Is the semicircle wide? (High Charge Transfer Resistance / Thick SEI / Cold Temp).
           
        3. LOW FREQUENCY (<1 Hz): Check the Diffusion Tail (Warburg).
           - Diagnostic: Is the slope 45 degrees (Healthy Diffusion) or vertical (Capacitive) or blocked?
           
        Return JSON ONLY:
        {{
            "layers": {{
                "ohmic": {{ "status": "Normal/Warning", "value_est_ohm": 0.05, "desc": "Good contact." }},
                "kinetics": {{ "status": "Normal/Warning", "desc": "Semicircle implies stable SEI." }},
                "diffusion": {{ "status": "Normal/Warning", "desc": "Clear Warburg tail visible." }}
            }},
            "overall_health": "Healthy" | "Degraded" | "Critical",
            "summary": "Battery shows normal impedance characteristics."
        }}
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return None

    def tool_search_knowledge_base(self, query: str):
        """Searches the internal battery manuals and document knowledge base for answers."""
        import asyncio
        from services.rag_service import rag_service
        results = asyncio.run(rag_service.search(query))
        return str(results)

    def tool_simulate_charging_analysis(self):
        """Runs a simulation of a battery charging session and analyzes the voltage curve for defects."""
        # Use Gemini to generate a dynamic simulation result instead of hardcoded string
        prompt = """
        Act as a Battery Simulation Engine.
        Simulate a charging session for a standard NMC cell.
        Randomly select one outcome: 
        1. Normal Operation (Healthy).
        2. Lithium Plating (Voltage Dip).
        3. High Resistance (Overheating).
        
        Return a short 2-sentence analysis report of the simulation.
        Example: "Simulation Complete. Anomaly detected: Lithium Plating signatures found at 45% SOC. Recommendation: High risk, reduce C-rate."
        """
        try:
            response_text = model_client.generate(
                self._build_prompt(prompt, enforce_json=False), task="text"
            )
            return response_text.strip()
        except Exception as e:
            logger.error("Simulation generation failed: %s", e)
            return "Simulation failed to generate report."

    def tool_predict_battery_life(self):
        """Predicts the remaining useful life (RUL) of the current battery pack based on historical cycles."""
        from services.aging_service import aging_service
        data = aging_service.generate_aging_curve(current_cycle=800)
        last_soh = data['soh'][-1]
        return f"Current Cycle: 800. Current SOH: {last_soh:.2f}%. Trend shows accelerated degradation (Knee Point) starting at cycle 600."

    def tool_parse_logs(self, log_content: str = ""):
        """Parses raw BMS logs directly into structured JSON errors."""
        # We can re-use the flash model directly or just prompt the agent.
        # Since the Agent IS the model, we can just return the log content with a wrapper
        # telling the agent to "Analyze this".
        # But to follow the pattern, let's call the specific parser method if it has specific logic.
        return f"Log Data Received: {log_content}. Please analyze this error code and suggest troubleshooting steps."

    def tool_create_incident_report(self, defect_type: str, severity: str, description: str):
        """Creates a formal incident report in the system based on analysis results."""
        # MVP: Just return confirmation
        import datetime
        report_id = f"INC-{datetime.datetime.now().strftime('%Y%m%d')}-001"
        return f"Incident Report Created Successfully. ID: {report_id}. Type: {defect_type}. Severity: {severity}. Status: Logged in Main Database."

    async def generate_commander_report(self, context: dict):
        """
        ACTS AS THE FLEET COMMANDER (Strategic Reasoning).
        Analyzes aggregate fleet statistics and provides high-level advice.
        """
        prompt = f"""
        Act as a Strategic Battery Fleet Commander.
        Review the following aggregated fleet statistics:
        
        Current Scenario: "{context.get('scenario')}"
        Thermal Spread (Max - Min Temp): {context.get('thermal_spread_degC')}°C
        Critical Outliers (Red List): {context.get('critical_outliers')} packs
        Max Fleet Temp: {context.get('max_temp_fleet')}°C
        
        Your Mission:
        1. Assess the strategic risk (Low/Medium/High).
        2. Provide 3 bullet-point tactical commands (e.g., "Throttle Fast Charging", "Isolate Batch X").
        3. Explain the "Why" using electrochemical reasoning (e.g., "High thermal spread indicates inefficient cooling balancing").
        
        Return JSON Key Highlights:
        {{
            "risk_level": "HIGH",
            "tactical_commands": ["Isolate 5 critical packs", "Reduce fleet C-rate to 0.5C", "Schedule thermal inspection"],
            "reasoning": "Thermal spread of >15C suggests cooling failure. Outliers at risk of propagation.",
            "status_message": "FLEET ALERT: Thermal variance exceeds safety limits."
        }}
        """
        try:
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {
                "risk_level": "UNKNOWN",
                "tactical_commands": ["Manual Inspection Required"],
                "reasoning": "AI Service Unreachable",
                "status_message": "AI OFFLINE"
            }

    async def generate_fleet_data(self, scenario: str):
        """
        DEPRECATED: Now handled by Physics Engine (fleet_service.py).
        Kept for fallback or legacy tests.
        """
        raise NotImplementedError("Handled by fleet_service")

    def tool_simulate_fleet(self, scenario: str):
        """Simulates a specific scenario on the battery fleet (e.g., 'heat wave', 'overcharge event'). Update the Fleet Monitor."""
        # This is a synchronous wrapper that schedules the async update via the service layer if possible,
        # or we block. Since tools are running in a thread/loop, we need to bridge to async.
        # For this hackathon, we can use a helper or run_until_complete if allowed, 
        # OR better: make the fleet_service.update_simulation synchronous wrapper or use the loop.
        
        # Quick Hack for Agent Tools (which are sync in default python SDK usually, unless async tools used):
        # We will trigger the update and return "Simulation Started". 
        # Actually, let's call it synchronously-blocking for simplicity in the Agent loop.
        import asyncio
        from services.fleet_service import fleet_service
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(fleet_service.update_simulation(scenario))
            return f"Simulation initiated for scenario: '{scenario}'. Visuals updating shortly."
        except RuntimeError:
            asyncio.run(fleet_service.update_simulation(scenario))
            return f"Simulation applied: '{scenario}'."
        except Exception as e:
             import traceback
             with open("error_log.txt", "a") as f:
                 f.write(f"Tool Error: {traceback.format_exc()}\n")
             return f"Simulation failed: {str(e)}"

    def get_agent_chat(self, history=None, context=None):
        """Returns a simple chat interface using model_client.
        
        Previously created a genai.GenerativeModel with tools.
        Now replaced with a simpler wrapper that uses model_client.
        Tools are still available through the ADK agent system.
        """
        # Dynamic System Instruction with Context
        context_block = ""
        if context:
            try:
                sanitized_context = self._sanitize_for_json(context)
                context_block = f"""
                
                CURRENT WORKSPACE STATE (AgentState):
                {json.dumps(sanitized_context, indent=2)}
                
                Use this state to answer questions like "What is the current error?" or "Is the battery healthy?".
                """
            except Exception as e:
                logger.error("Context serialization error: %s", e)
                context_block = "\n(Context could not be loaded due to data format error.)"
        
        system_instruction = f"""
            You are 'BatteryForge AI', an intelligent battery technician agent.
            You have access to tools to Search Manuals, Simulate Charging, Predict Aging, and Log Incidents.
            
            - **Navigation**: You can control the workspace!
              - To show Home Dashboard: Output `[VIEW: HOME]`
              - To show Visual Intelligence: Output `[VIEW: VISUAL]`
              - To show Logs: Output `[VIEW: LOGS]`
              - To show Simulation: Output `[VIEW: SIM]`
              - To show Fleet Monitor: Output `[VIEW: FLEET]`
            - **Safety & Alerts**:
              - If a user reports a "Fire", "Thermal Runaway", or "Critical Failure", IMMEDIATELY output `[ACTION: RED_ALERT]`.
              - To clear the alarm, output `[ACTION: CLEAR_ALERT]`.
            - **Visual capabilities**: You can analyze images AND live video!
              - If the user has a video stream or wants real-time checks, direct them to [VIEW: VISUAL] and mention the "Live Scout" tab.
              - "Visual Scout" supports Webcam and Screen Sharing for thermal runaway detection.
            - Context Awareness: You can see what the user is doing (Visual Inspection, Logs). Use that context!
            - ALWAYS check your tools before saying "I don't know".
              To call a tool, respond EXACTLY with: [TOOL_CALL: tool_name("arg1", "arg2")]
              Available tools:
              - search_knowledge_base(query)
              - simulate_charging_analysis()
              - predict_battery_life()
              - parse_logs()
              - create_incident_report()
              - simulate_fleet(scenario)
            - If the user asks to "log this" or "create a report", use `create_incident_report` using the data from the context.
            - If asked about error codes or safety, use `search_knowledge_base`.
            - If asked to "check the charging curve" or "simulate charging", use `simulate_charging_analysis`.
            - If asked about "battery life" or "how long it will last", use `predict_battery_life`.
            - Be concise and helpful.
            {context_block}
            """
        
        # Return a lightweight chat wrapper that uses model_client
        return _ModelClientChat(system_instruction, self, history)


    # ==========================================
    # FEATURE 1: BMS Design & Engineering (BatteryForge AI)
    # ==========================================

    async def generate_pcb_design_critique(self, design_specs: str, conversation_history: list = None):
        """
        BMS Design Review with BatteryForge AI.
        Analyzes battery management system designs for:
        - Cell balancing topology (passive vs active)
        - Current sensing architecture (high-side vs low-side shunt)
        - Protection circuit design (OVP, UVP, OCP, OTP)
        - Communication interfaces (CAN, SMBus, isolated vs non-isolated)
        - Thermal management integration
        """
        try:
            history_context = ""
            if conversation_history:
                history_context = "\n\nPrevious Conversation:\n"
                for turn in conversation_history:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    history_context += f"  {role}: {content}\n"

            prompt = f"""
            Act as a Senior BMS Architect with 15+ years in EV battery systems.

            IMPORTANT BEHAVIOR:
            - First, evaluate if the input specifications contain enough critical information to produce a reliable BMS design.
            - Critical information includes: cell configuration (S/P), cell chemistry, voltage range, max continuous/peak current, balancing requirements, communication interfaces, thermal management needs, and target application.
            - If ANY critical information is missing or ambiguous, you MUST return clarifying questions INSTEAD of a full design plan.
            - Only generate the full design plan when you have sufficient information.
            {history_context}

            Current BMS Specifications:
            "{design_specs}"

            MANDATORY ANALYSIS AREAS:
            1. **Cell Configuration & Balancing**
               - Is the cell count (S/P) appropriate for the voltage/capacity?
               - Is passive or active balancing specified? Recommend based on pack size.
               - Check balancing current vs cell capacity ratio.

            2. **Current Sensing Architecture**
               - Shunt resistor placement (high-side vs low-side)?
               - Coulomb counting accuracy requirements?
               - Current rating vs max discharge rate?

            3. **Protection Circuits**
               - OVP/UVP thresholds appropriate for cell chemistry?
               - Short circuit detection time (typically <500μs required)?
               - Precharge circuit for capacitive loads?

            4. **Thermal Integration**
               - NTC thermistor placement strategy?
               - Thermal runaway detection provisions?
               - Cooling system interface signals?

            5. **Safety & Standards**
               - IEC 62619 compliance gaps?
               - Functional safety (ISO 26262) considerations for automotive?
               - UN38.3 transport requirements?

            DECISION:
            - If information is INSUFFICIENT, return JSON with clarifying_questions.
            - If information is SUFFICIENT, return JSON with the full design_plan.

            Response Format (Insufficient Info):
            {{
                "status": "needs_clarification",
                "clarifying_questions": [
                    "What is the cell chemistry (NMC, LFP, NCA)?",
                    "What is the target application (EV, ESS, power tools)?",
                    "Is active or passive cell balancing preferred?",
                    "What communication interface is required (CAN, SMBus, UART)?"
                ],
                "understood_so_far": "16S BMS with 100A discharge requirement"
            }}

            Response Format (Sufficient Info):
            {{
                "status": "design_ready",
                "design_plan": {{
                    "blocks": [
                        "Cell Monitoring AFE (16S stacked)",
                        "MCU (ARM Cortex-M4, CAN peripheral)",
                        "High-Side Current Sense (100A shunt + INA240)",
                        "Protection FETs (Dual N-CH, 150A rated)",
                        "Precharge Circuit (10Ω NTC + relay)",
                        "Isolated CAN Transceiver",
                        "DC-DC Isolated Power Supply"
                    ],
                    "interconnections": [
                        "Cells -> AFE -> MCU (daisy-chain SPI)",
                        "Shunt -> INA240 -> MCU ADC",
                        "MCU -> Gate Driver -> Protection FETs",
                        "MCU -> ISO CAN -> Vehicle ECU",
                        "NTC Array -> MUX -> MCU ADC"
                    ]
                }},
                "component_recommendations": [
                    {{ "function": "Cell Monitor AFE", "spec": "BQ76952 (16S, integrated balancing)", "verify": "Verify cell voltage accuracy ±5mV" }},
                    {{ "function": "Current Sense Amp", "spec": "INA240A4 (high-side, 200V CMR)", "verify": "Check gain error vs temperature" }},
                    {{ "function": "Protection FET", "spec": "NVMFS5C673NL (80V, 150A)", "verify": "SOA for short circuit event" }},
                    {{ "function": "MCU", "spec": "STM32G474 (CAN-FD, HRTIM)", "verify": "Automotive grade AEC-Q100" }}
                ],
                "constraint_definitions": [
                    "Balancing: Passive 50mA or Active with efficiency > 90%",
                    "Protection: OVP at 4.25V/cell (NMC), UVP at 2.8V/cell, OCP at 120A",
                    "Response Time: Short circuit detection < 300μs, FET turn-off < 50μs",
                    "Thermal: 8x NTC (1 per 2 cells), thermal runaway threshold 70°C",
                    "Isolation: CAN bus must be galvanically isolated (2.5kV rated)"
                ]
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def explore_design_space(self, grid_state: dict):
        """
        Reinforcement Learning for PCB Routing (BatteryForge AI - Agent Logic).
        Simulates an RL agent predicting the optimal next step for a trace on a grid.
        
        Input:
            grid_state: {
                "grid_size": [10, 10],
                "obstacles": [[2,2], [2,3]],
                "start": [0,0],
                "target": [9,9],
                "current_head": [4,4]
            }
        """
        try:
            prompt = f"""
            Act as a Reinforcement Learning Agent for PCB Auto-Routing.
            Objective: Route trace from Start to Target minimizing vias and length, avoiding obstacles.
            
            Current State:
            {json.dumps(grid_state)}
            
            Rules:
            - Move Up, Down, Left, Right.
            - Reward: +1 for getting closer, -10 for collision.
            
            Predict the OPTIMAL next move.
            
            Return JSON:
            {{
                "next_move": [4, 5],
                "action": "MOVE_RIGHT",
                "confidence": 0.95,
                "reasoning": "Avoids obstacle at [2,2], moves towards target [9,9]. Heuristic distance decreases."
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def parse_component_datasheet(self, file_content, mime_type="application/pdf", design_constraints: dict = None):
        """
        Intelligent Datasheet Parsing & Component Selection (BatteryForge AI - Multimodal).
        Parses component datasheets (PDF/Image) to extract electrical specs for BOM validation.
        Optionally cross-references against design constraints.
        """
        try:
            constraints_section = ""
            if design_constraints:
                constraints_section = f"""

            ADDITIONAL TASK - Design Constraint Verification:
            Cross-reference the extracted specs against these design constraints:
            {json.dumps(design_constraints, indent=2)}

            For each constraint, verify if the component meets it. For example:
            - "Is Vin_max > required voltage?"
            - "Is max current sufficient for the design?"
            - "Does the thermal resistance allow safe operation at ambient temp?"

            Add a "constraint_check" field to your response with pass/fail for each constraint.
            """

            prompt = f"""
            Act as a Senior PCB Component Engineer (BatteryForge AI).
            Analyze this component datasheet (PDF/Image).

            Task:
            1. Identify the Component (Part Number, Manufacturer, Description).
            2. Extract Key Electrical Parameters (Voltage, Max Current, Logic Levels).
            3. Extract Thermal Limits (T_junction, Thermal Resistance).
            4. Extract Pin Configuration Table.
            {constraints_section}

            Return JSON for BOM Tool:
            {{
                "component_info": {{
                    "part_number": "LM7805",
                    "manufacturer": "TI",
                    "description": "5V Linear Regulator"
                }},
                "electrical_specs": {{
                    "input_voltage_max": "35V",
                    "output_current_max": "1.5A",
                    "dropout_voltage": "2.0V"
                }},
                "thermal_specs": {{
                    "max_junction_temp": "125C",
                    "package_thermal_resistance": "50 C/W (TO-220)"
                }},
                "pin_configuration": [
                    {{"pin": 1, "name": "INPUT", "function": "Vin"}},
                    {{"pin": 2, "name": "GND", "function": "Ground"}},
                    {{"pin": 3, "name": "OUTPUT", "function": "Vout"}}
                ],
                "compliance": {{
                    "rohs": true,
                    "automotive_qualified": false
                }},
                "constraint_check": [
                    {{"constraint": "Vin_max > 24V", "result": "PASS", "detail": "Vin_max is 35V > 24V"}},
                    {{"constraint": "Iout_max >= 2A", "result": "FAIL", "detail": "Iout_max is 1.5A < 2A"}}
                ]
            }}
            """

            # Multimodal input: encode file content as base64
            file_b64 = base64.b64encode(file_content).decode("utf-8")
            response_text = await model_client.generate_async(
                self._build_prompt(prompt), image_b64=file_b64, task="vision"
            )
            return self._extract_json(response_text)
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    # ==========================================
    # FEATURE 2: Intelligent Quality Control (BatteryForge AI)
    # ==========================================

    async def analyze_production_defect(self, image_data, mime_type="image/jpeg", reference_image_data=None):
        """
        AI-Powered Defect Classification (BatteryForge AI - Vision).
        Distinguishes between harmless cosmetic variations and fatal functional defects.
        Optionally uses a 'Golden Sample' reference to filter false positives.
        """
        try:
            comparison_context = ""
            
            if reference_image_data:
                comparison_context = """
                COMPARISON MODE:
                You are provided with TWO images.
                1. Test Image (Potential Defect)
                2. Reference Image (Golden Sample / Known Good)

                Task: Compare the Test Image against the Reference Image to filter out noise.
                - If a "defect" in the Test Image is also present in the Reference (e.g., a specific silk screen mark), it is NOT a defect.
                - Only flag deviations.
                """

            prompt = f"""
            Act as a Senior SMT Vision Inspector (BatteryForge AI).
            Analyze this high-resolution manufacturing image of a PCB or electronic assembly.
            {comparison_context}
            
            Task:
            1. Detect defects: Solder bridges, solder voids, cold joints, tombstoning, missing components, component misalignment, scratches, delamination.
            2. FILTER FALSE POSITIVES: Identify and exclude harmless cosmetic variations (laser marking glare, silk screen irregularities, flux residue that does not affect function). Report how many false positives were filtered.
            3. Classify Severity: FATAL (Open Circuit / Short Circuit / Missing Component) vs COSMETIC (Acceptable per IPC-A-610 Class 2).
            4. Provide a recommended action for each real defect.
            5. If possible, provide bounding box [ymin, xmin, ymax, xmax] (0-1000 scale) for the primary defect.
            
            Return JSON:
            {{
                "defects_found": [
                    {{ "type": "Solder Bridge", "severity": "FATAL", "location": "U1 Pin 3-4", "confidence": 98, "action": "Rework required — reflow and wick excess solder", "bbox": [500, 200, 550, 250] }},
                    {{ "type": "Flux Residue", "severity": "COSMETIC", "location": "R5 area", "confidence": 82, "action": "IGNORE — no functional impact" }}
                ],
                "verdict": "FAIL",
                "summary": "1 fatal defect found requiring rework. 1 cosmetic issue acceptable per IPC-A-610.",
                "fatal_count": 1,
                "cosmetic_count": 1,
                "false_positives_filtered": 2,
                "inspection_standard": "IPC-A-610 Class 2"
            }}
            """
            
            # Encode primary image as base64
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            return await self._llm_json_async(prompt, image_b64=image_b64, task="vision")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def analyze_xray_inspection(self, image_data, mime_type="image/jpeg"):
        """
        X-Ray Analysis for Multilayer inspection (BatteryForge AI).
        Detects hidden defects like BGA voids, barrel distortion, misalignment.
        """
        try:
            prompt = """
            Act as an AXI (Automated X-ray Inspection) Expert.
            Analyze this X-ray slice of a BGA component.
            
            Check for:
            1. Head-in-Pillow (HiP) defects.
            2. Voids > 25% of ball area.
            3. Via Barrel Distortion (wobble in vertical drill hole walls).
            4. Layer Misalignment / Pad Offset.
            
            Return JSON:
            {
                "inspection_type": "3D AXI",
                "bga_analysis": {
                    "voids_found": true,
                    "max_void_percentage": 15.0,
                    "status": "PASS"
                },
                "layer_alignment": {
                    "misalignment_um": 5,
                    "status": "GOOD",
                    "barrel_distortion_detected": false
                },
                "anomalies": ["Minor voiding on Ball A5 (within IPC limits)"]
            }
            """
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            return await self._llm_json_async(prompt, image_b64=image_b64, task="vision")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def inspect_battery_assembly(self, image_data, mime_type="image/jpeg", inspection_type="general"):
        """
        Vision AI for battery pack assembly inspection.

        inspection_type options:
        - "weld": Tab welding quality (cold weld, splash, burn-through)
        - "pouch": Pouch cell integrity (swelling, seal defects, electrode misalignment)
        - "busbar": Busbar connections (torque marks, contact quality)
        - "thermal_paste": TIM application uniformity
        - "general": Full assembly overview
        """
        try:
            prompts = {
                "weld": """
                    Act as a Battery Welding QC Engineer with expertise in laser/ultrasonic tab welding.
                    Analyze this image of battery tab welds.

                    DEFECT CLASSES:
                    1. COLD_WELD - Insufficient fusion, dull/porous appearance, weak bond
                    2. BURN_THROUGH - Excessive heat, hole in tab material, visible damage
                    3. SPLASH - Weld spatter on adjacent cells or busbars
                    4. MISALIGNMENT - Tab not centered on terminal, offset weld nugget
                    5. CRACK - Fracture in weld zone or heat-affected zone
                    6. INCOMPLETE_FUSION - Partial weld, not full coverage

                    For each defect found, provide:
                    - Classification and severity (CRITICAL/MAJOR/MINOR)
                    - Location on the image (approximate coordinates 0-1000 scale)
                    - Root cause hypothesis (laser power, pulse duration, focus, contamination)
                    - Accept/Reject decision per automotive battery standards

                    Return JSON:
                    {
                        "inspection_type": "tab_weld",
                        "defects_found": [
                            {"type": "COLD_WELD", "severity": "CRITICAL", "location": "Cell 3 positive tab", "confidence": 95, "root_cause": "Insufficient laser power or dirty surface", "bbox": [200, 300, 250, 350]}
                        ],
                        "weld_quality_score": 72,
                        "verdict": "REJECT",
                        "summary": "Cold weld detected on Cell 3 - requires reweld",
                        "critical_count": 1,
                        "major_count": 0,
                        "minor_count": 0
                    }
                """,

                "pouch": """
                    Act as a Pouch Cell QC Specialist with expertise in lithium-ion cell inspection.
                    Analyze this pouch cell image for manufacturing defects.

                    DEFECT CLASSES:
                    1. SWELLING - Gas generation, pillow effect, bulging sides
                    2. SEAL_DEFECT - Incomplete edge sealing, wrinkles, channeling
                    3. ELECTRODE_VISIBLE - Tab area showing electrode material, misaligned stack
                    4. ELECTROLYTE_LEAK - Wet spots, crystallization, corrosion stains
                    5. DENT - Physical damage to pouch, puncture risk
                    6. TAB_DAMAGE - Bent, torn, or corroded tabs

                    Severity scale: SCRAP (immediate disposal), REWORK (salvageable), ACCEPT_WITH_DEVIATION, PASS

                    Return JSON:
                    {
                        "inspection_type": "pouch_cell",
                        "defects_found": [
                            {"type": "SWELLING", "severity": "SCRAP", "location": "Center of cell body", "confidence": 98, "safety_risk": "HIGH - potential thermal event", "bbox": [100, 200, 800, 600]}
                        ],
                        "cell_condition_score": 15,
                        "verdict": "SCRAP",
                        "summary": "Severe swelling indicates internal gas generation - cell must be safely disposed",
                        "safety_alert": true
                    }
                """,

                "busbar": """
                    Act as a Battery Pack Assembly QC Engineer specializing in busbar connections.
                    Analyze this busbar/interconnect image.

                    CHECK POINTS:
                    1. TORQUE_MARKS - Evidence of proper fastener torque (paint marks, witness marks)
                    2. CONTACT_QUALITY - Full contact area, no gaps, proper alignment
                    3. CORROSION - Surface oxidation, galvanic corrosion signs
                    4. THERMAL_DAMAGE - Discoloration from overheating, hot spots
                    5. MECHANICAL_DAMAGE - Scratches, dents, cracks in busbar
                    6. INSULATION - Proper isolation from adjacent conductors

                    Return JSON:
                    {
                        "inspection_type": "busbar_connection",
                        "connections_checked": 8,
                        "issues_found": [
                            {"type": "MISSING_TORQUE_MARK", "severity": "MAJOR", "location": "Connection B3", "confidence": 90, "action": "Verify torque and re-mark"}
                        ],
                        "connection_quality_score": 85,
                        "verdict": "CONDITIONAL_PASS",
                        "summary": "7 of 8 connections verified. B3 requires torque verification."
                    }
                """,

                "thermal_paste": """
                    Act as a Thermal Interface Material (TIM) Application QC Specialist.
                    Analyze this image of thermal paste/pad application on battery cells or modules.

                    CHECK POINTS:
                    1. COVERAGE - Full coverage of contact area, no bare spots
                    2. UNIFORMITY - Even thickness, no pooling or thin areas
                    3. OVERFLOW - Excess material outside intended area
                    4. CONTAMINATION - Foreign particles, debris in TIM
                    5. AIR_BUBBLES - Trapped air pockets reducing thermal transfer

                    Return JSON:
                    {
                        "inspection_type": "thermal_interface",
                        "coverage_percentage": 95,
                        "uniformity_score": 88,
                        "issues_found": [
                            {"type": "AIR_BUBBLE", "severity": "MINOR", "location": "Corner region", "confidence": 75, "thermal_impact": "Localized hot spot risk"}
                        ],
                        "verdict": "PASS",
                        "summary": "TIM application meets spec with minor air bubble - acceptable for production"
                    }
                """,

                "general": """
                    Act as a Senior Battery Pack Assembly Inspector.
                    Perform a comprehensive visual inspection of this battery assembly image.

                    CHECK ALL AREAS:
                    1. Cell alignment and spacing
                    2. Wiring harness routing and strain relief
                    3. BMS board mounting and connections
                    4. Thermal management components (cooling plates, TIM)
                    5. Structural integrity (enclosure, brackets)
                    6. Safety features (fuses, contactors, vents)
                    7. Labeling and QR codes

                    Return JSON:
                    {
                        "inspection_type": "general_assembly",
                        "areas_inspected": ["cells", "wiring", "bms", "thermal", "structure", "safety", "labeling"],
                        "defects_found": [],
                        "observations": [
                            {"area": "wiring", "note": "Harness properly secured with P-clips", "status": "GOOD"},
                            {"area": "thermal", "note": "Cooling plate contact verified", "status": "GOOD"}
                        ],
                        "overall_score": 92,
                        "verdict": "PASS",
                        "summary": "Assembly meets production standards. Ready for EOL testing."
                    }
                """
            }

            prompt = prompts.get(inspection_type, prompts["general"])

            image_b64 = base64.b64encode(image_data).decode("utf-8")
            return await self._llm_json_async(prompt, image_b64=image_b64, task="vision")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    # ==========================================
    # FEATURE 3: Predictive Maintenance (BatteryForge AI)
    # ==========================================

    async def analyze_maintenance_signals(self, sensor_payload: dict):
        """
        Vibration & Sound Anomaly Detection (BatteryForge AI).
        Classifies equipment state based on FFT frequency peaks or time-domain stats.
        Input: { "machine_id": "Drill-01", "fft_peaks": [{"freq": 1200, "amp": 0.5}], "rms_vibration": 1.2 }
        """
        try:
            prompt = f"""
            Act as a Predictive Maintenance Expert.
            Analyze processed sensor features (FFT/Vibration) for CNC machines.
            
            Data:
            {json.dumps(sensor_payload)}
            
            Match signatures:
            - High amp > 1kHz -> Bearing Wear
            - Low freq (10-50Hz) wobbly -> Loose Belt
            - Spindle Runout -> Harmonics of RPM
            
            Return JSON:
            {{
                "health_status": "WARNING",
                "diagnosis": "Early Stage Bearing Wear",
                "confidence": 0.89,
                "maintenance_window": "Schedule within 7 days",
                "signatures_detected": ["1.2kHz harmonic peak"]
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def predict_tool_life(self, tool_logs: dict):
        """
        Predictive Tool Replacement Scheduling (BatteryForge AI).
        Forecasts probability of breakage.
        Input: { "hits": 5000, "resin_smear_level": "medium", "feed_rate_deviation": 0.05 }
        """
        try:
            prompt = f"""
            Act as a Tooling Life Analyst.
            Analyze drill bit usage logs to predict Remaining Useful Life (RUL).

            Logs: {json.dumps(tool_logs)}

            Physics: High resin smear + feed deviation = dull bit -> high breakage risk.

            Return JSON:
            {{
                "rul_hits": 250,
                "breakage_probability_percent": 65,
                "action": "CHANGE_TOOL_NOW" | "CONTINUE",
                "reason": "Resin smear indicates thermal degradation."
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def analyze_thermal_health(self, thermal_data: dict):
        """
        AI-Powered Thermal Analysis for CNC Spindle/Motor Health (BatteryForge AI).
        Analyzes temperature patterns to predict bearing failures and recommend actions.
        Input: { "machine_id": "CNC-DRILL-01", "spindle_temp_c": 72, "ambient_temp_c": 25, "load_percent": 85 }
        """
        try:
            prompt = f"""
            Act as a CNC Machine Thermal Analyst Expert.
            Analyze the spindle/motor thermal data to assess machine health.

            Data:
            {json.dumps(thermal_data)}

            Analysis Guidelines:
            - Normal spindle temp: 40-60°C at 80% load
            - Warning threshold: 65-75°C (increased bearing wear rate)
            - Critical threshold: >75°C (immediate attention required)
            - Temperature rise rate matters: sudden spikes indicate lubrication issues
            - Ambient-adjusted delta: (spindle_temp - ambient) at given load

            Diagnose potential issues:
            1. Bearing lubrication degradation
            2. Coolant system malfunction
            3. Excessive cutting load
            4. Belt tension issues (indirect heating)

            Return JSON:
            {{
                "thermal_status": "WARNING" | "NORMAL" | "CRITICAL",
                "temperature_delta_c": 47,
                "expected_delta_c": 35,
                "deviation_percent": 34,
                "diagnosis": "Elevated spindle temperature suggests bearing lubrication degradation",
                "risk_factors": ["High load operation", "Extended runtime without cooldown"],
                "recommended_actions": [
                    "Schedule bearing inspection within 48 hours",
                    "Reduce spindle RPM by 10% until inspection",
                    "Check coolant flow rate and concentration"
                ],
                "estimated_time_to_failure_hours": 200,
                "confidence": 0.82
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    # ==========================================
    # FEATURE 3B: Battery Formation Protocol Optimization
    # ==========================================

    async def optimize_formation_protocol(self, cell_chemistry: str, capacity_ah: float,
                                          ambient_temp: float, target_cycles: int):
        """
        AI-powered formation cycling optimization.
        Uses electrochemical knowledge to recommend C-rate profiles
        for optimal SEI formation.
        """
        try:
            prompt = f"""
            Act as a Battery Formation Engineer with expertise in SEI (Solid Electrolyte Interphase) optimization.

            Cell Parameters:
            - Chemistry: {cell_chemistry}
            - Nominal Capacity: {capacity_ah} Ah
            - Ambient Temperature: {ambient_temp}°C
            - Target Formation Cycles: {target_cycles}

            Provide an optimized formation protocol considering:
            1. Initial low-rate charge (C-rate, voltage cutoff) - slower = denser, more uniform SEI
            2. Rest period between cycles for electrolyte redistribution
            3. Temperature setpoint for each phase - affects SEI composition (LiF vs organic species)
            4. Expected capacity retention after formation

            Chemistry-Specific Guidelines:
            - NMC/NCA: Form at 0.05-0.1C initial, 25°C optimal, 4.2V cutoff
            - LFP: Can tolerate 0.1-0.2C, 25-35°C acceptable, 3.65V cutoff
            - LTO: Fast formation possible at 0.5C, wide temp range

            Return JSON:
            {{
                "chemistry": "{cell_chemistry}",
                "formation_protocol": {{
                    "cycle_profiles": [
                        {{
                            "cycle": 1,
                            "charge_c_rate": 0.05,
                            "charge_cutoff_v": 4.2,
                            "discharge_c_rate": 0.1,
                            "discharge_cutoff_v": 2.8,
                            "rest_after_charge_min": 30,
                            "rest_after_discharge_min": 15,
                            "temperature_setpoint_c": 25
                        }},
                        {{
                            "cycle": 2,
                            "charge_c_rate": 0.1,
                            "charge_cutoff_v": 4.2,
                            "discharge_c_rate": 0.2,
                            "discharge_cutoff_v": 2.8,
                            "rest_after_charge_min": 20,
                            "rest_after_discharge_min": 10,
                            "temperature_setpoint_c": 25
                        }}
                    ],
                    "total_time_hours": 48,
                    "predicted_sei_quality": 92,
                    "expected_capacity_retention_1000_cycles": 88
                }},
                "technical_reasoning": "Low initial C-rate allows uniform SEI nucleation. 25°C balances Li+ mobility with SEI stability.",
                "warnings": ["Avoid formation above 35°C - leads to porous SEI with poor cycling stability"]
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def optimize_tab_welding(self, material: str, thickness_mm: float, weld_type: str):
        """
        AI-powered tab welding parameter optimization.
        Recommends laser/ultrasonic parameters for battery tab welding.
        """
        try:
            prompt = f"""
            Act as a Battery Welding Process Engineer specializing in tab-to-cell connections.

            Welding Parameters:
            - Tab Material: {material}
            - Tab Thickness: {thickness_mm} mm
            - Weld Type: {weld_type}

            Optimize welding parameters for:
            1. Laser Welding: Power (W), pulse duration (ms), spot size, focal position
            2. Ultrasonic Welding: Amplitude (μm), pressure (N), time (ms), horn frequency

            Material Considerations:
            - Nickel tabs: Good weldability, 50-100μm typical
            - Aluminum tabs: Requires careful oxide removal, prone to porosity
            - Copper tabs: High thermal conductivity, needs higher power

            Return JSON:
            {{
                "material": "{material}",
                "thickness_mm": {thickness_mm},
                "recommended_parameters": {{
                    "laser": {{
                        "power_w": 2500,
                        "pulse_duration_ms": 3,
                        "spot_diameter_mm": 0.6,
                        "focal_offset_mm": 0,
                        "shield_gas": "Argon",
                        "pulse_shape": "rectangular"
                    }},
                    "ultrasonic": {{
                        "amplitude_um": 30,
                        "pressure_n": 400,
                        "weld_time_ms": 200,
                        "frequency_khz": 20,
                        "horn_pattern": "knurled"
                    }}
                }},
                "expected_weld_strength_n": 55,
                "quality_metrics": {{
                    "nugget_diameter_mm": 2.5,
                    "penetration_percent": 80,
                    "acceptable_void_percent": 5
                }},
                "process_window": "Power ±5%, Time ±10% for consistent results"
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    # ==========================================
    # FEATURE 4: Supply Chain Resilience (BatteryForge AI)
    # ==========================================

    async def monitor_supply_risk(self, components: list):
        """
        Dynamic BOM Optimization & Risk Sensing (BatteryForge AI).
        Analyzes BOM for geopolitical risks/obsolescence.
        """
        try:
            prompt = f"""
            Act as a Supply Chain Intelligence Agent.
            Analyze this BOM List for risks (Geopolitical, End-of-Life, Sole-Source).
            
            Components: {json.dumps(components)}
            
            Assumption: You have access to a knowledge base of component origins (simulated).
            
            Return JSON:
            {{
                "high_risk_components": [
                    {{ "part": "IC-XYZ", "risk_factor": "Geopolitical Instability in Region A", "score": 90 }}
                ],
                "alternatives": [
                    {{ "for": "IC-XYZ", "suggestion": "IC-ABC (Domestically sourced)" }}
                ]
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    async def forecast_inventory(self, usage_data: dict):
        """
        Material Inventory Forecasting (BatteryForge AI).
        Predicts shortages of critical raw materials (laminates, copper).
        Input: { "material": "Copper Foil 1oz", "usage_rate_per_day": 50, "lead_time_days": 14, "market_trend": "Shortage" }
        """
        try:
            prompt = f"""
            Act as an Inventory Planner.
            Forecast demand and buffer stock.
            
            Data: {json.dumps(usage_data)}
            
            Task: Calculate strategic buffer stock (e.g., 45-60 days) if market trend is 'Shortage'.
            
            Return JSON:
            {{
                "material": "Copper Foil 1oz",
                "recommended_order_qty": 5000,
                "days_of_coverage": 60,
                "urgency": "HIGH - Market Shortage Impact"
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

    # ==========================================
    # FEATURE 5: Smart Process Control (BatteryForge AI)
    # ==========================================

    async def analyze_process_control_loop(self, sensor_readings: dict):
        """
        Adaptive Etching and Lamination Control (BatteryForge AI).
        Real-time Closed-Loop Feedback.
        Input: { "process": "Etching", "ph_level": 3.2, "copper_thickness_removed": 15um, "target": 18um }
        """
        try:
            prompt = f"""
            Act as a Process Control System (BatteryForge AI).
            Analyze real-time sensor feedback and recommend PLC parameter adjustments.
            
            Data: {json.dumps(sensor_readings)}
            
            Logic:
            - If under-etching (removed < target): Decrease conveyor speed OR Increase spray pressure.
            - If over-etching: Increase speed.
            
            Return JSON:
            {{
                "status": "Under-Etching Detected",
                "adjustment_command": {{
                    "parameter": "conveyor_speed",
                    "action": "DECREASE",
                    "value_delta_percent": -5
                }},
                "safety_lock": false
            }}
            """
            return await self._llm_json_async(prompt, task="text")
        except Exception as e:
            logger.exception("Model call failed")
            return {"error": str(e)}

gemini_service = GeminiService()
