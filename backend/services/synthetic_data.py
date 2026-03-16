from services.model_client import model_client
import logging

logger = logging.getLogger("synthetic_data_service")

class SyntheticDataService:

    async def generate_defect_image(self, defect_type="swelling"):
        """
        Generate a synthetic lithium-ion battery defect image description.
        """

        prompt = f"""
        Generate a photorealistic close-up image of a lithium-ion pouch cell battery.
        The battery should exhibit clear signs of '{defect_type}'.
        The defect should be seamlessly blended into the surface (similar to Poisson editing).
        Technical industrial lighting, high resolution.
        """

        try:
            response = await model_client.generate_async(prompt, task="text")

            return {
                "status": "generated",
                "description": response,
                "data": None
            }

        except Exception as e:
            logger.error("Generation error: %s", e)
            return {"error": str(e)}