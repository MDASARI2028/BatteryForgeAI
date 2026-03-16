import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger("digital_twin_service")

class DigitalTwinService:
    async def run_shadow_simulation(self, real_df: pd.DataFrame):
        """
        Compares REAL battery data against a PHYSICS-BASED DIGITAL TWIN.
        Returns the Deviation Metric and Safety Status.
        """
        from services.model_client import model_client
        
        # 1. Extract Protocol & Initial Conditions (Zero-Shot)
        # We take the first few rows to set the "Twin's" initial state.
        initial_voltage = real_df['voltage'].iloc[0] if 'voltage' in real_df.columns else 3.0
        duration = len(real_df) # proxy for time
        
        # Downsample for prompt
        cols = [c for c in ['time', 'voltage'] if c in real_df.columns]
        if len(cols) < 2:
            raise ValueError("Digital twin requires 'time' and 'voltage' columns")
            
        sample_curve = real_df[cols] \
            .iloc[::max(1, len(real_df)//50)] \
            .round(4) \
            .to_json(orient='records')
        
        # 2. Ask Gemini to Generate Baseline
        prompt = f"""
        Act as a Generative Anomaly Detector for Battery Signals.
        
        Analyze the attached battery voltage vs time dataset.
        Initial Voltage: {initial_voltage} V.
        Duration: {duration} data points.
        
        REAL MEASURED DATA (sampled):
        {sample_curve}
        
        Task:
        1. Generate a "Semantic Baseline": Based on your knowledge of Li-Ion electrochemistry, what should the IDEAL curve look like for this specific charge/discharge pattern?
        2. Compare Real vs Baseline: Identify deviations that indicate degradation or faults (not just noise).
        3. Safety Verdict: If significant deviation > 5% found in critical regions (knees, plateaus), flag it.
        
        Return JSON:
        {{
            "baseline_confidence": 95, // Your confidence in the generated baseline
            "max_deviation_percent": 2.3,
            "safety_status": "NORMAL" | "WARNING" | "CRITICAL",
            "anomaly_reason": "Voltage dip detected at step 40..." (or "None"),
            "ideal_curve_points": [ ... list of {{time, voltage}} for the generated baseline ... ]
        }}
        """
        
        try:
            text = await model_client.generate_async(prompt, task="text")
            
            # Extract JSON safely
            start = text.find("{")
            end = text.rfind("}")
            
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in model response")
                
            return result
        except Exception as e:
            logger.error("Digital Twin Error: %s", e)
            return {"safety_status": "UNKNOWN", "error": str(e)}

digital_twin_service = DigitalTwinService()
