"""
Thermal Runaway Prediction Service

Uses Gemini 3 Vision to analyze battery thermal images/video and predict
thermal runaway events with countdown timer.
"""

import io
import numpy as np
import base64
import json
from datetime import datetime, timedelta
from services.model_client import model_client

class ThermalRunawayPredictor:
    def __init__(self):
        """Initialize the thermal runaway prediction service"""
        # Thermal safety thresholds (celsius)
        self.thresholds = {
            'normal': 45,
            'elevated': 60,
            'warning': 80,
            'critical': 100,
            'runaway': 120
        }
    
    def analyze_thermal_image(self, image_data: bytes, current_temp: float = None):
        """
        Analyze thermal image for runaway indicators
        
        Args:
            image_data: Image bytes (thermal or RGB)
            current_temp: Optional current temperature reading
            
        Returns:
            dict with prediction results
        """
        try:
            
            # RunPod Vision Analysis (using ModelClient)
            prompt = f"""
            Analyze the attached thermal image of a lithium-ion battery cell.
            
            Current Temperature: {current_temp}°C (if provided)
            
            Analyze for thermal runaway indicators:
            1. **Hotspot Detection**: Identify localized heating zones
            2. **Temperature Gradient**: Calculate temperature variation across cell
            3. **Thermal Runaway Probability**: Estimate risk (0-100%)
            4. **Time to Critical Event**: If runaway likely, estimate minutes until thermal runaway
            5. **Root Cause**: Likely failure mechanism (internal short, dendrites, damage)
            
            Respond in this exact JSON format:
            {{
                "risk_level": "normal|elevated|warning|critical|imminent",
                "probability": 0-100,
                "hotspot_detected": true/false,
                "hotspot_temp_estimate": <celsius>,
                "time_to_runaway_minutes": <number or null>,
                "failure_mechanism": "description",
                "recommended_action": "immediate action required",
                "confidence": 0-100
            }}
            """
            
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            try:
                result_text = model_client.generate(prompt, image_b64=image_b64, task="vision")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Model inference failed: {str(e)}"
                }
            
            # Extract JSON safely
            start = result_text.find("{")
            end = result_text.rfind("}")
            
            if start != -1 and end != -1:
                json_str = result_text[start:end+1]
                prediction = json.loads(json_str)
            else:
                return {
                    'success': False,
                    'error': "No JSON object found in model response"
                }
                
            valid_risks = {"normal", "elevated", "warning", "critical", "imminent"}
            if prediction.get("risk_level") not in valid_risks:
                prediction["risk_level"] = "warning"
            
            # Add metadata
            prediction['analysis_timestamp'] = datetime.now().isoformat()
            prediction['current_temp'] = current_temp
            
            # Calculate countdown if applicable
            if prediction.get('time_to_runaway_minutes'):
                eta = datetime.now() + timedelta(minutes=prediction['time_to_runaway_minutes'])
                prediction['estimated_runaway_time'] = eta.isoformat()
            
            return {
                'success': True,
                'prediction': prediction
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_video_stream(self, frames: list, frame_temps: list = None):
        """
        Analyze sequence of thermal frames to detect trends
        
        Args:
            frames: List of image bytes
            frame_temps: Optional list of temperature readings
            
        Returns:
            dict with trend analysis
        """
        try:
            # Analyze last frame
            last_frame = frames[-1]
            current_temp = frame_temps[-1] if frame_temps else None
            
            result = self.analyze_thermal_image(last_frame, current_temp)
            
            if not result['success']:
                return result
            
            # Calculate temperature trend if multiple readings
            if frame_temps and len(frame_temps) >= 3:
                recent_temps = frame_temps[-5:]  # Last 5 readings
                temp_trend = np.polyfit(range(len(recent_temps)), recent_temps, 1)[0]
                
                result['prediction']['temperature_trend_c_per_sec'] = round(temp_trend, 3)
                
                # If rapid heating detected, adjust time to runaway
                if temp_trend > 0.5:  # >0.5°C/sec is concerning
                    result['prediction']['risk_level'] = 'critical'
                    if current_temp:
                        # Estimate time to reach 120°C at current rate
                        time_to_120 = (120 - current_temp) / (temp_trend * 60)
                        result['prediction']['time_to_runaway_minutes'] = max(1, int(time_to_120))
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_safety_report(self, prediction: dict):
        """
        Generate human-readable safety report
        
        Args:
            prediction: Prediction dict from analyze_thermal_image
            
        Returns:
            str: Formatted safety report
        """
        risk = prediction.get('risk_level', 'unknown')
        prob = prediction.get('probability', 0)
        action = prediction.get('recommended_action', 'Monitor situation')
        
        report = f"""
╔══════════════════════════════════════════════════════════
║ 🔥 THERMAL RUNAWAY SAFETY REPORT
╚══════════════════════════════════════════════════════════

RISK ASSESSMENT:
  • Risk Level: {risk.upper()}
  • Probability: {prob}%
  • Confidence: {prediction.get('confidence', 0)}%

THERMAL ANALYSIS:
  • Current Temperature: {prediction.get('current_temp', 'N/A')}°C
  • Hotspot Detected: {'YES ⚠️' if prediction.get('hotspot_detected') else 'NO'}
  • Hotspot Temperature: {prediction.get('hotspot_temp_estimate', 'N/A')}°C

FAILURE MECHANISM:
  {prediction.get('failure_mechanism', 'Unknown')}

⏰ TIME TO CRITICAL EVENT:
  {prediction.get('time_to_runaway_minutes', 'N/A')} minutes

🚨 RECOMMENDED ACTION:
  {action}

Analysis Timestamp: {prediction.get('analysis_timestamp', 'N/A')}
        """
        
        return report.strip()


# Singleton instance
thermal_predictor = ThermalRunawayPredictor()
