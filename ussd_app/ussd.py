from flask import Flask, request
import os
import sys


from dotenv import load_dotenv

sys.path.insert(1, '/')

from send_sms import chunk_message
from ai_response import nutrients_metrics_func, crop_recommendations_func, future_forecast_func, market_projection_func
from payment import initiate_payment


app = Flask(__name__)

@app.route("/ussd", methods=['POST'])
def ussd():
    session_id   = request.values.get("sessionId", None)
    serviceCode  = request.values.get("serviceCode", None)
    phone_number = request.values.get("phoneNumber", None)
    text         = request.values.get("text", "default")

    user_response = text.split("*")

    if text == "":
        response = "CON Welcome to Bombadier \n"
        response += "1. IOT Kit Services\n"
        response += "2. Loan Services\n"
        response += "3. Report Fraud/Complaint\n"
        response += "4. About Our Company"

    



    # ----------------------------- #
    # SECTION 1: IOT SOIL KIT Services #
    # ----------------------------- #
    elif text == "1":
        # if text == "1":
        response = "CON IOT Kit Services \n"
        response += "1. Purchase IOT Kit\n"
        response += "2. My kit Dashboard\n"
        response += "3. Report Faulty\n"
        response += "4. Subscribe for daily notifications"

    elif text == "1*1":
        response = f"CON {initiate_payment(phone_number)}"

    elif text == "1*2":
        response = "CON My Dashboard \n"
        response += "1. Realtime Nutrient Metrics \n"
        response += "2. Recommendations \n"
        response += "3. Future Forecast \n"
        response += "4. Market Projection \n"

    elif text == "1*2*1":
        response = f"END You will recieve a message shortly {chunk_message(phone_number, nutrients_metrics_func())}"

    elif text == "1*2*2":
        response = f"END You will recieve a message shortly {chunk_message(phone_number, crop_recommendations_func())}"


    elif text == "1*2*3":
        response = f"END You will recieve a message shortly {chunk_message(phone_number, future_forecast_func())}"


    elif text == "1*2*4":
        response = f"END You will recieve a message shortly {chunk_message(phone_number, market_projection_func())}"


    elif text == "1*3":
        response = "CON Report Faulty \n"
        response += "Call Customer Care \n"
        response += "Send Kit Serial Number \n"

    



    # ----------------------------- #
    # SECTION 2: Loan Services      #
    # ----------------------------- #

    elif text == "2":
        response = "CON Loan Services \n"
        response += "1. Apply for a Loan\n"  
        response += "2. Check Loan Eligibility\n"  
        response += "3. Loan Balance & Repayment\n"  
        response += "4. Farm Input Loans\n"  
        response += "5. Produce / Harvest Loans\n"  
        response += "6. Insurance & Risk Cover\n"  
        response += "7. Farmer Profile & Credit Score\n"  
        response += "8. Contact / Support\n" 

    elif text == "2*1":
        response = "CON Loan Application \n"
        resposne += "1. Instant Cash Loan\n"  
        resposne += "2. Farm Input Loan (Seeds, Fertilizer, Pesticides)\n"  
        resposne += "3. Equipment Loan (Tractor, Irrigation Kit, etc.)\n"  
        resposne += "4. Group/Cooperative Loan \n" 
        resposne += "5. Back \n" 


    elif text == "2*2":
        response = "CON Check Loan Eligibility \n"
        response += "1. Check Eligibility Status \n"  
        response += "2. View Credit Score \n"  
        response += "3. Update Farm Information \n"  
        response += "4. View Loan Limit \n"  
        response += "5. Back \n" 

    elif text == "2*3":
        response = "CON Loan Balance \n"
        response += "1. View Outstanding Balance \n"  
        response += "2. Make a Repayment (via Mobile Money) \n"  
        response += "3. View Repayment History \n"  
        response += "4. Request Statement (SMS) \n"  
        response += "5. Back \n"

    elif text == "2*4":
        response = "CON Farm Inpu Loans\n"
        response += "1. Request Seeds Loan \n"  
        response += "2. Request Fertilizer Loan \n"  
        response += "3. Request Pesticides Loan \n"  
        response += "4. Check Input Loan Status \n"  
        response += "5. Back \n"  


    elif text == "2*5":
        response = "CON Produce / Harvest Loans \n"
        response += "1. Request Advance on Expected Harvest \n"  
        response += "2. Link to Buyer/Market \n"  
        response += "3. Track Produce Delivery \n"  
        response += "4. Repay After Sale \n"  
        response += "5. Back \n"  


    elif text == "2*6":
        response = "CON Insurance & Risk Cover \n"
        response += "1. Enroll in Crop Insurance \n"  
        response += "2. Enroll in Livestock Insurance \n"  
        response += "3. Check Insurance Status \n"  
        response += "4. File a Claim \n"  
        response += "5. Back \n"  


    elif text == "2*7":
        response = "CON Credit Score \n"
        response += "1. Update Farm Details (Size, Type, Crop) \n"  
        response += "2. View My Credit Score \n"  
        response += "3. Link to Cooperative \n"  
        response += "4. Add Next of Kin \n"  
        response += "5. Back \n"  


    elif text == "2*8":
        response = "CON Get Help\n"
        response += "1. Talk to a Loan Officer \n"  
        response += "2. FAQs \n"  
        response += "3. Report a Problem \n"  
        response += "4. Back \n"  


            
    # ----------------------------- #
    # SECTION 3: Report Fraud     #
    # ----------------------------- #

    elif text == "3":
        response = "CON File a Claim\n"
        response += "1. Talk to Customer Care"






    # ----------------------------- #
    # SECTION 4: About the Company    #
    # ----------------------------- #

    elif text == "4":
        response = "END File a Claim \n"
        





    return response



if __name__ == '__main__':
    app.run(debug=True, port="8000")
