import africastalking
import os

from dotenv import load_dotenv


load_dotenv()

africastalking.initialize(
		username="EMID",
		api_key=os.getenv("AT_API_KEY")

	)


sms = africastalking.SMS


def chunk_message(phone_number, message):
	recipients = [f"{str(phone_number)}"]


	sender = "Test"

	try:
		response = sms.send(message, recipients, sender)

		print(response)

	except Exception as e:
		print(f'Houston! we have a problem: {e}')
