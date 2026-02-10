# from vessel_manager import VesselSpecificManager

# # Create manager
# manager = VesselSpecificManager()

# # Get vessel instance  
# vessel = manager.get_vessel_instance("12345")
# print(f"Created vessel: {vessel.imo}")


import comtypes.client

# Generate the SpeechLib wrapper if not already generated
comtypes.client.GetModule('C:\\Windows\\System32\\Speech\\Common\\sapi.dll')

# Now import the generated SpeechLib
from comtypes.gen import SpeechLib
