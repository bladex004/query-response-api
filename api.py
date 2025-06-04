from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI-Driven Query Response API")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-Driven Query Response API. Visit /docs for API documentation."}

class QueryRequest(BaseModel):
    query: str

class CustomAI:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.training_data = {"queries": [], "cluster_labels": [], "generated_responses": {}}
        self.data_file = os.getenv("DATA_FILE", "training_data.json")
        self.query_count = 0
        self.train_interval = 5
        self.used_response_sets = {}
        self.num_clusters = 10
        self.response_formats = [
            "To {query}, please use the website’s designated process.",
            "For {query}, follow the instructions on our site.",
            "Regarding {query}, check the appropriate section on the website.",
            "To address {query}, refer to the site’s guidelines.",
            "For assistance with {query}, explore the available options online.",
            "To handle {query}, use the provided tools on the website.",
            "Concerning {query}, navigate to the site’s resources."
        ]
        self.query_response_map = {
            "greeting": {
                "keywords": ["hello", "hi", "help"],
                "responses": [
                    "Hello and welcome to Flix! I’m here to assist with {query}.",
                    "Greetings and welcome to Flix. How may I help with {query}?",
                    "Welcome to Flix! My name is here to assist with {query}.",
                    "Hi, Welcome to Flix, I am here to assist with {query}.",
                    "I appreciate that you have reached out to us with your query about {query}.",
                    "Thank you for reaching out with your inquiry about {query}.",
                    "I am grateful that you contacted us regarding your concern about {query}.",
                    "I appreciate that you are reporting to us about this concern: {query}. I will look into it immediately."
                ]
            },
            "verification": {
                "keywords": ["verify", "booking details", "pnr"],
                "responses": [
                    "Sure, I'm here to help. To assist you effectively with {query}, please provide the following details: Booking number or PNR Number, Passenger's full name, Booking email address or booking phone number.",
                    "Certainly, I'm ready to assist you with {query}. To ensure I can help you effectively, please provide the following details: Booking number, Passenger's full name, Booking email address or booking phone number.",
                    "Could you please provide the following details to help us complete the verification process for {query}?",
                    "To proceed with the verification for {query}, may I kindly ask you to share the required information?",
                    "For the purpose of {query}, please help us by providing the details listed on our website.",
                    "Thank you for sharing the details for {query}. Please allow me a moment to look for your concern.",
                    "Thank you for providing the information for {query}. Please allow me a moment to check with the details.",
                    "Thank you for sharing these details on {query}. Just a moment, please, I'm checking.",
                    "I'm still checking your booking/ride details for {query}. Please be with me.",
                    "I got your booking but need to check the issue with your ride for {query}."
                ]
            },
            "appreciate_patience": {
                "keywords": ["patience", "wait"],
                "responses": [
                    "Thank you so much for your patience with {query}.",
                    "I really appreciate your patience with {query}.",
                    "Thank you so much for notifying me about the issue: {query}.",
                    "Thank you for reaching out to me about this: {query}.",
                    "I will help resolve your issue with {query}.",
                    "I appreciate your patience in this matter: {query}.",
                    "Your patience is appreciated for {query}.",
                    "Thank you for your patience regarding {query}."
                ]
            },
            "change_boarding_point": {
                "keywords": ["change boarding point", "board from another location"],
                "responses": [
                    "May I know why you want to change the boarding point for {query}?",
                    "As per our T&C, your ticket is valid for the boarding location booked, and I regret to inform that boarding from another location isn’t possible for {query}.",
                    "I understand your preference for boarding the bus from your desired location for {query}. However, to ensure a smooth process, we kindly ask you to board from the designated location where the QR code can be scanned by the bus staff.",
                    "While I appreciate your desire to board the bus from a different location for {query}, please note that the process requires the QR code to be scanned by the host at the specified boarding point.",
                    "I understand your request to board the bus from your chosen location for {query}. Unfortunately, due to our procedures, the QR code must be scanned by the host at the specified point.",
                    "Thank you for your understanding. Although we recognize your preferred boarding location for {query}, our process mandates that the QR code be scanned by the host at the assigned area.",
                    "I appreciate your interest in boarding from your preferred spot for {query}. However, to adhere to our process, the QR code must be scanned by the host at the designated boarding area."
                ]
            },
            "boarding_point_details": {
                "keywords": ["boarding point details", "where to board"],
                "responses": [
                    "The ticket contains essential details such as the boarding points, bus route number, a link in the bottom right corner, and a GPS link that can be accessed by clicking on the boarding point for more information about {query}.",
                    "You can find important information on the ticket, including boarding points, the bus route number, a link at the bottom right corner, and a GPS link accessible by clicking on the boarding point for {query}.",
                    "The ticket provides various details, such as the boarding points, the bus route number, a link located in the bottom right corner, and a GPS link by clicking on the boarding point for further information about {query}."
                ]
            },
            "price_difference": {
                "keywords": ["price difference", "why price change"],
                "responses": [
                    "I would like to inform you that prices are dynamic and may change on the website with time for {query}.",
                    "Please note that our prices are dynamically adjusted based on demand, availability, and other factors to ensure the best possible experience for all our passengers for {query}.",
                    "To provide you with an accurate and fair pricing, our rates are dynamically updated for {query}. We recommend booking early to secure the best available price.",
                    "We always recommend our passengers a price lock feature where the price shown at the time of selection is reserved for 10 minutes, allowing passengers to complete their booking at the same rate for {query}.",
                    "We suggest booking early and utilizing the price lock feature to secure the most favorable fare for {query}.",
                    "Different boarding or drop-off locations, timing, seat type, extra services, or demand can affect pricing for {query}."
                ]
            },
            "pre_departure_call": {
                "keywords": ["pre-departure call", "call before departure"],
                "responses": [
                    "I completely understand that you are expecting a call from the bus staff for {query}. Please note that pre-departure calls are not mandatory; however, the host might call you before arriving at your departure point."
                ]
            },
            "bus_running_late": {
                "keywords": ["bus running late", "late bus"],
                "responses": [
                    "I regret to inform you that the bus is running late due to some operational reason and apologize for the inconvenience caused due to delay for {query}.",
                    "So requesting you to please be available on the boarding point as the ride has already been departed from the previous boarding point and now going to reach your boarding point as quickly as possible for {query}.",
                    "As per the updated information, the expected time of the ride to reach your boarding point is available on our website for {query}.",
                    "Additionally, I’m sharing the bus host contact number for your better assistance with {query}.",
                    "I hope I’m able to help you with your query: {query}. Is there anything else I may help you?"
                ]
            },
            "missed_bus": {
                "keywords": ["missed bus", "left behind"],
                "responses": [
                    "We see you as a valued customer and do not wish to leave you behind for {query}. However, it’s necessary to move the bus on schedule, as we have a commitment to punctuality.",
                    "We appreciate you as a valued customer and have no intention of leaving you behind for {query}. At the same time, we must adhere to the bus's scheduled departure time for the sake of punctuality.",
                    "We truly value you as a customer, and it’s not in our interest to leave you behind for {query}. However, we also have to ensure the bus departs on time, as punctuality is essential.",
                    "I sincerely apologize for the inconvenience caused by missing the bus for {query}.",
                    "I understand how frustrating this can be, and I’m here to help you with any further assistance you may need for {query}.",
                    "Please help me with the booking reference number or the PNR number for verification process so that I can help you with the information associated with your journey with Flix for {query}.",
                    "I would like to inform you that the ride has departed from the designated boarding point as per the mentioned time of the ticket and all the other passengers from the same boarding point have taken their rides without facing any difficulties for {query}.",
                    "I can understand that due to some reason you are not being able to reach the boarding point before the mentioned time due to which you are unable to take the ride for {query}.",
                    "After a careful investigation, I must conclude that as there are no operational issues associated with your ride for {query}, I apologize and I’m unable to provide any alternative nor process with the refund.",
                    "I completely understand your concern, and I genuinely want to help you with {query}. However, the system currently doesn’t allow me to process the refund."
                ]
            },
            "bus_host_number": {
                "keywords": ["bus number", "host number"],
                "responses": [
                    "I really apologize that I am unable to provide the bus number; however, you may identify your ride through the route number mentioned on the ticket for {query}."
                ]
            },
            "bus_driver_host_details": {
                "keywords": ["bus driver details", "host details"],
                "responses": [
                    "I regret to inform you that we don't have access to the bus driver or host's contact information for {query}. So, we always request our passengers to be at the pickup point 15 minutes prior to the departure."
                ]
            },
            "where_is_my_bus": {
                "keywords": ["where is my bus", "bus location"],
                "responses": [
                    "As checked, the ride is scheduled, and it will arrive at the boarding point as per the mentioned time on the ticket for {query}.",
                    "I’m requesting you to please be present at the boarding point 15 minutes before the departure time of the bus for {query}, so that you can easily board the bus.",
                    "Here I’m sharing the bus route number, which is mentioned on the ticket and available in front of the bus for {query}.",
                    "Additionally, I’m sharing the tracking link of your ride for {query}, which will be operational when the ride departs from your designated boarding point."
                ]
            },
            "where_is_boarding_point": {
                "keywords": ["where is boarding point", "boarding location"],
                "responses": [
                    "This is the address of your boarding point for {query}, available on the ticket.",
                    "This is the Google map link for the exact boarding point location for {query}.",
                    "I’m requesting you to please be available on the boarding point 15 minutes before the departure time of the bus for {query}, so that you can easily board the bus."
                ]
            },
            "bus_delay": {
                "keywords": ["bus delay", "delayed bus"],
                "responses": [
                    "I’m sincerely apologizing for the delay of the ride and sorry for the inconvenience caused to you for {query}.",
                    "I would like to inform you that due to some operational reason, the ride has been delayed for {query}, and our operational team is trying to manage the delay.",
                    "I regret to inform you that due to heavy traffic, the ride got stuck for {query}, and now it’s back on track.",
                    "Here I’m sharing the bus route number for {query}, which is mentioned on the ticket and available in front of the bus.",
                    "This is the tracking link of the ride for {query}, which will be operational when the bus starts from the initial boarding point.",
                    "So, it’s a request to please be available on the boarding point with your all the boarding details as the ride is going to reach your boarding point anytime for {query}."
                ]
            },
            "pax_running_late": {
                "keywords": ["running late", "wait for me"],
                "responses": [
                    "Extremely sorry to inform you that the bus will depart from the boarding point as per the scheduled time for {query}, so requesting you to please try to reach the boarding point before the mentioned time.",
                    "Unfortunately, the bus cannot wait for our delayed passengers for {query}.",
                    "If you can’t reach in time for {query}, you can cancel your ride up to 15 minutes before departure via manage my booking on our website.",
                    "This is the bus host contact number and route number of your bus for {query}, mentioned on your ticket."
                ]
            },
            "ride_cancellation": {
                "keywords": ["ride cancellation", "cancel ride"],
                "responses": [
                    "I’m really sorry for the inconvenience caused to you due to the ride cancellation for {query}, and a self-help link has been provided to you via email.",
                    "So requesting you to please check your email inbox along with the spam folder for {query}, and after clicking on that link, you will be able to book an alternative ride completely free of cost.",
                    "Or if you want, I can help you with the full ticket refund by cancelling the ticket from our end for {query}, and the refund will be credited within 7 working days.",
                    "Thank you for the confirmation. Please give a moment to proceed for the cancellation process for {query}.",
                    "This is the cancellation invoice for your ticket for {query}.",
                    "After the completion of the given time, if you are facing any difficulties regarding the refund for {query}, then don’t hesitate to reach out to us."
                ]
            },
            "pax_no_show": {
                "keywords": ["no show", "missed ride refund"],
                "responses": [
                    "I deeply apologize for the inconvenience you’re experiencing for {query}, and I understand your situation. I’m unable to proceed with either the refund or booking an alternative ride.",
                    "I understand your concern and want to assist you with {query}, but unfortunately, our system prevents us from taking any further action in this case.",
                    "I understand your situation, and I’d likely feel frustrated too for {query}. I am sorry I am unable to meet your request at this time.",
                    "As per our company policy, in such cases where the service has been provided as intended for {query}, we are unable to process a refund.",
                    "Following a detailed investigation, we confirmed that the bus arrived at the designated boarding point, other passengers boarded successfully, and the bus departed as scheduled for {query}.",
                    "After a thorough review of the incident, we have determined that the bus arrived at the designated boarding point on time for {query}, and other passengers boarded without issue.",
                    "We have thoroughly reviewed the situation and found that the bus arrived at the designated boarding point for {query}, successfully boarded other passengers, and departed after the scheduled time.",
                    "Upon review, we found that the bus reached the boarding point on time for {query}, boarded other passengers, and departed after the scheduled time."
                ]
            },
            "changes_after_booking_denial": {
                "keywords": ["change booking", "modify booking"],
                "responses": [
                    "Once your ticket is booked, we are unable to modify it from our side for {query}. Please visit our website, go to ‘Manage My Booking,’ and fill in the required information.",
                    "We cannot change your booking after it has been confirmed for {query}. Please go to our website, select ‘Manage My Booking,’ and provide the needed details."
                ]
            },
            "booking_process": {
                "keywords": ["booking process", "how to book"],
                "responses": [
                    "To assist with {query}, please access the booking links provided on our website and select 'CONTINUE' to proceed to the checkout page.",
                    "For {query}, ensure you have a valid identification proof like Aadhar Card, Passport, or Driving License for boarding.",
                    "Regarding {query}, we accept payment methods like Credit Cards, UPI, and Net Banking, with a platform fee of Rs 5.",
                    "For {query}, note that seat options include standard, window, and premium seats, with details available on the website.",
                    "To complete {query}, fill out the necessary details for seat reservation, passenger information, contact details, and payment on our booking portal."
                ]
            },
            "changes_via_mmb": {
                "keywords": ["change date", "change time", "postpone ride"],
                "responses": [
                    "If you wish to change the date or time of your ride for {query}, cancel, or postpone it, you can easily make these adjustments through the ‘Manage My Booking’ section.",
                    "If you need to reschedule, cancel, or postpone your ride for {query}, you can manage these changes through the ‘Manage My Booking’ portal."
                ]
            },
            "complaint_feedback": {
                "keywords": ["complaint", "feedback"],
                "responses": [
                    "Thank you so much! We're thrilled to hear that you enjoyed your experience with us for {query}.",
                    "Your feedback on my response for {query} is important to me! Please provide it via our website.",
                    "We appreciate your feedback for {query} and look forward to hearing from you on our support page."
                ]
            },
            "rude_staff_complaint": {
                "keywords": ["rude driver", "rude host", "staff behavior"],
                "responses": [
                    "I sincerely apologize for the unpleasant experience you had with the driver and the bus host for {query}.",
                    "We deeply regret that their behavior was not up to the standards you expect and deserve for {query}.",
                    "We value our passengers’ comfort and respect, and it’s truly disappointing to hear about this situation for {query}.",
                    "Please be assured that I will escalate this matter to the relevant team for a thorough review and appropriate action for {query}.",
                    "I would also be grateful if you could provide feedback on how I’ve assisted you today with {query}."
                ]
            },
            "bus_breakdown_refund": {
                "keywords": ["breakdown refund", "bus issue refund"],
                "responses": [
                    "Thank you for reaching out, and I sincerely apologize for the inconvenience you’ve experienced due to the breakdown of the bus for {query}.",
                    "To assist you further with {query}, could you please provide your booking reference number or PNR number, along with the email address or phone number?",
                    "We need your patience and cooperation in this matter for {query}. Our team is actively working to get updates from the host.",
                    "While we understand your request for a refund or alternative ride for {query}, we are currently in the process of resolving the situation.",
                    "Please rest assured that we are committed to providing you with the best support possible for {query}.",
                    "Thank you for your understanding and cooperation for {query}."
                ]
            },
            "route_details": {
                "keywords": ["route details", "bus route"],
                "responses": [
                    "I regret to inform you that we don’t have the specific route information about the ride for {query}; however, we have the access for the stop locations.",
                    "These are the stop locations associated with your existing booking for {query}, available on our website."
                ]
            },
            "change_ticket_date": {
                "keywords": ["change ticket date", "reschedule date"],
                "responses": [
                    "Yes, you can change the date of your journey up to 15 minutes before departure time for {query} via the manage my booking section on our website.",
                    "Please note that the prices are dynamic in nature for {query}, and any fare difference will be displayed during the rescheduling process.",
                    "If the new fare is lower for {query}, then the difference will be refunded to your original payment method within 7 working days."
                ]
            },
            "route_information": {
                "keywords": ["route information", "ride route"],
                "responses": [
                    "I regret to inform you that we don’t have the route information of the ride for {query}; however, we have the access for the stop location associated with that booking.",
                    "If you have already booked the ticket and want to know the route information of your ride for {query}, you may click on the link provided on our website."
                ]
            },
            "flix_lounge": {
                "keywords": ["flix lounge", "anand vihar lounge"],
                "responses": [
                    "Thank you for reaching out to us. We apologize for any confusion, but please note that the Flix Lounge facility is not available at the Anand Vihar location for {query}.",
                    "We understand your concern, and we sincerely apologize for any inconvenience caused for {query}. Since the Anand Vihar location doesn't have a Flix Lounge, we suggest waiting at the boarding point."
                ]
            },
            "bus_delay_less_120": {
                "keywords": ["delay less than 120", "short delay"],
                "responses": [
                    "I’m really sorry for the delay of the bus for {query}. I understand this can be frustrating, and I sincerely apologize for the inconvenience caused.",
                    "As checked, the bus was delayed due to some operational reasons and traffic issues for {query}.",
                    "According to our T&C, we can only offer a refund if the bus is delayed by more than 120 minutes for {query}.",
                    "Please let me know if you have any other questions about {query}. We truly appreciate your patience."
                ]
            },
            "bus_delay_more_120": {
                "keywords": ["delay more than 120", "long delay"],
                "responses": [
                    "I sincerely apologize for the delay for {query}. I understand how frustrating this can be.",
                    "Upon investigation, I can confirm that the bus is delayed by more than 2 hours for {query} due to operational reasons.",
                    "If you prefer not to wait, I can proceed with cancelling your ticket and initiate a refund for {query}.",
                    "Your refund has been initiated, and the amount will be credited to your source account within 7 working days for {query}.",
                    "Thank you for your understanding and patience with {query}."
                ]
            },
            "bus_breakdown_ac": {
                "keywords": ["ac not working", "bus breakdown"],
                "responses": [
                    "I sincerely apologize for the inconvenience caused due to the bus breakdown and as the AC not working for {query}.",
                    "I’ve already highlighted this issue to our team for {query}, and they are working on resolving it as soon as possible.",
                    "As the ride has not been cancelled for {query}, I regret to inform you that we cannot process a refund at this moment.",
                    "Please stay assured that the team is working hard to address the issue for {query}, and we are here to support you throughout.",
                    "Thank you for your understanding and patience with {query}."
                ]
            },
            "luggage_policy": {
                "keywords": ["luggage", "baggage"],
                "responses": [
                    "Thank you for reaching out regarding our luggage policy for {query}.",
                    "I’m happy to inform you that you are allowed to bring 7kg of hand luggage and 20kg of regular luggage completely free of charge for {query}.",
                    "Additionally, you may bring one extra luggage item of 20kg per passenger for {query}. Since space is limited, we recommend booking your additional luggage early.",
                    "For more details, please visit our luggage policy on our website for {query}."
                ]
            },
            "cancel_ticket": {
                "keywords": ["cancel ticket", "cancellation refund"],
                "responses": [
                    "I would like to inform you that you may cancel your ticket for {query} up to 15 minutes before the departure time via our website.",
                    "You can cancel it through our website’s ‘Manage My Booking’ section for {query}.",
                    "If you choose the voucher option for {query}, the refund amount will be generated in the form of voucher and sent via email.",
                    "If you select the option for cash refund for {query}, it will be credited to your source account within 7 working days excluding weekends and holidays.",
                    "To know more about our cancellation policy for {query}, please follow the link provided on our website."
                ]
            },
            "stranded_passenger": {
                "keywords": ["stranded", "left behind"],
                "responses": [
                    "We’re very sorry to hear about your situation for {query} and understand how frustrating this must be.",
                    "Could you please share your booking reference number, registered email, and phone number so we can look into this for {query}?",
                    "Unfortunately, we are unable to offer a refund or arrange an alternative ride in this situation for {query}, as per our company policy.",
                    "We kindly request that you make alternative travel arrangements to reach the destination for {query}."
                ]
            },
            "lost_item": {
                "keywords": ["lost item", "left behind item"],
                "responses": [
                    "We’re very sorry to hear that your belongings were left on the bus for {query}.",
                    "To assist you in recovering your items for {query}, please fill out our Lost and Found form on our website.",
                    "Our team will connect with you as soon as we have any updates for {query}."
                ]
            },
            "travel_with_pet": {
                "keywords": ["travel with pet", "pet policy"],
                "responses": [
                    "Thank you for your inquiry regarding traveling with pets for {query}.",
                    "Unfortunately, at this time, we are unable to accommodate pets on our buses for {query}."
                ]
            },
            "prices_discounts": {
                "keywords": ["prices", "discounts"],
                "responses": [
                    "The price shown on your ticket is the final price for {query}. You do not need to pay any further amount.",
                    "I apologize, but currently, there are no offers or discounts available for {query}.",
                    "I would like to inform you that prices are dynamic in nature and may change on the website with time for {query}.",
                    "Please note that our prices are dynamically adjusted based on demand, availability for {query}.",
                    "Our pricing structure is designed to reflect real-time demand and availability for {query}."
                ]
            },
            "blanket_service": {
                "keywords": ["blanket service", "blankets"],
                "responses": [
                    "I regret to inform you but as of now we are not providing blankets on board for {query}; however, we recommend our customers to carry one along.",
                    "We're pleased to know that blankets and water bottles have been provided for your convenience on all rides for {query}.",
                    "For your comfort, we have ensured blankets are available on every ride for {query}.",
                    "We're happy to let you know that blankets have been provided for all passengers on each ride for {query}.",
                    "To enhance your journey, blankets have been made available on all rides for {query}."
                ]
            },
            "water_bottle_service": {
                "keywords": ["water bottle", "water service"],
                "responses": [
                    "We regret to inform you that, as of now, we do not offer water bottle services on our Flix buses for {query}.",
                    "We recommend that passengers bring their own water bottles for their journey for {query}."
                ]
            },
            "washroom_service": {
                "keywords": ["washroom", "toilet"],
                "responses": [
                    "I regret to inform you that as of now, Flix does not provide washroom facilities on the bus for {query}. However, the bus host will take care of comfort breaks."
                ]
            },
            "seat_change": {
                "keywords": ["change seat", "seat assignment"],
                "responses": [
                    "We apologize, but we are unable to change your seat for {query} as it is automatically assigned.",
                    "Unfortunately, we cannot change your seat for {query} because seats are auto-assigned.",
                    "I'm sorry, but seat changes are not possible for {query} as they are assigned automatically.",
                    "Regrettably, we cannot accommodate seat change requests for {query} as seats are auto-assigned.",
                    "Please note that seats are automatically assigned by the system for {query}."
                ]
            },
            "shadow_booking": {
                "keywords": ["shadow booking", "booking error"],
                "responses": [
                    "It's sad to hear about the inconvenience caused by {query}. Please provide passenger's full name, email ID, phone number, and a screenshot of the payment transaction.",
                    "I apologize for the inconvenience, but I couldn't locate any booking matching the details provided for {query}.",
                    "I regret to inform you that I am unable to locate any booking with the provided details for {query}."
                ]
            },
            "no_refund_statement": {
                "keywords": ["no refund", "refund denial"],
                "responses": [
                    "After thoroughly investigating the incident, we found that the bus arrived at the designated boarding point and other passengers successfully boarded the bus for {query}. Unfortunately, we are unable to process a refund."
                ]
            },
            "refund_7_days": {
                "keywords": ["refund processing", "refund time"],
                "responses": [
                    "Your ticket has been cancelled as of today for {query}. It will take up to 7 working days for the amount to be credited back to your account.",
                    "We would like to inform you that your ticket has been cancelled for {query}. The refund amount will be processed within 7 working days.",
                    "Your ticket has been cancelled for {query}. It will take up to 7 working days for the refund to appear in your account.",
                    "We would like to inform you that your ticket was cancelled for {query}. The refund amount will be processed within 7 working days."
                ]
            },
            "refund_tat_crossed": {
                "keywords": ["refund not received", "refund delay"],
                "responses": [
                    "We would like to inform you that the refund has been initiated from our end for {query}. Please check with your bank regarding the status.",
                    "This is to notify you that the refund was processed from our side for {query}. Please verify with your bank."
                ]
            },
            "closing": {
                "keywords": ["bye", "thank you", "end"],
                "responses": [
                    "It appears that our communication has paused for {query}. If you require further assistance, please feel free to initiate a new chat session via our support page.",
                    "Our communication seems to have temporarily halted for {query}. Should you need additional help, please start a new chat session.",
                    "It appears there's a pause in our communication for {query}. If you need more assistance, please begin a new chat session.",
                    "Our communication has paused momentarily for {query}. If you require further assistance, please initiate a new chat session.",
                    "Thank you for contacting Flix for {query}. Have a great day.",
                    "I’m happy to have assisted you with your inquiry for {query}! If you have any other questions, please reach out.",
                    "It was a pleasure assisting you today with {query}. If you need further assistance, don't hesitate to contact us.",
                    "I'm glad I could help you with your inquiry for {query}. If there's anything else you need, please let me know.",
                    "It was my pleasure to assist you with your inquiry for {query}. If you have any more questions, just let me know.",
                    "I’m pleased I could assist you today with {query}! Wishing you a fantastic day ahead!"
                ]
            },
            "request_rating": {
                "keywords": ["rate", "survey", "feedback"],
                "responses": [
                    "Looking forward to your valuable feedback towards my response for {query}, the link will be there right after the chat ends.",
                    "We appreciate your feedback for {query} and look forward to hearing from you. The option will be available once our chat concludes.",
                    "Your feedback towards my response for {query} is important for me! The link will be provided after our conversation ends.",
                    "Please do provide me with your valuable feedback for {query}, which will help me to boost my morale and assist other passengers.",
                    "I’d love to hear your thoughts for {query}. Your feedback means a lot to me and will inspire me to assist others better.",
                    "I did my best to help you with {query}! If you have any feedback, I’d really appreciate it.",
                    "Your thoughts mean a lot to me for {query}, so if you have any feedback, please share.",
                    "If you found my chat helpful for {query}, I would greatly appreciate it if you could rate your experience via a survey.",
                    "We value your feedback for {query} and would love to hear your thoughts. Feel free to share any comments.",
                    "I'm pleased that I could assist you with {query}. Your feedback is invaluable to us via a survey after this chat.",
                    "I’m happy to have assisted you with {query}. A survey will be sent to gather your insights.",
                    "It’s great to know I could assist you with {query}! You’ll receive a survey after our chat to share your thoughts.",
                    "It would mean a lot if you could complete the customer satisfaction survey for {query}.",
                    "It will be great if you could fill out the customer satisfaction survey for {query}.",
                    "Thank you for allowing me to assist you with {query}. A survey link will be sent after our chat.",
                    "I am grateful for the opportunity to assist you with {query}. Your feedback via the survey link would mean a great deal."
                ]
            }
        }
        self.load_training_data()

    def load_training_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "queries" in data and "cluster_labels" in data:
                        self.training_data = data
                        if "generated_responses" not in self.training_data:
                            self.training_data["generated_responses"] = {}
                        logger.info(f"Loaded training data from {self.data_file}")
                    else:
                        logger.warning(f"Invalid data format in {self.data_file}, initializing empty")
                        self.training_data = {"queries": [], "cluster_labels": [], "generated_responses": {}}
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {self.data_file}: {e}")
                self.training_data = {"queries": [], "cluster_labels": [], "generated_responses": {}}
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                self.training_data = {"queries": [], "cluster_labels": [], "generated_responses": {}}
        else:
            logger.info(f"No training data file found at {self.data_file}, initializing empty")
            self.training_data = {"queries": [], "cluster_labels": [], "generated_responses": {}}

    def save_training_data(self):
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"Saved training data to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def custom_cluster(self, features):
        if features.shape[0] < self.num_clusters:
            return [-1] * features.shape[0]
        indices = random.sample(range(features.shape[0]), self.num_clusters)
        centroids = np.array([features[i].toarray()[0] for i in indices])
        labels = [-1] * features.shape[0]
        max_iterations = 50
        for _ in range(max_iterations):
            new_labels = []
            for i in range(features.shape[0]):
                distances = np.linalg.norm(features[i].toarray() - centroids, axis=1)
                new_labels.append(np.argmin(distances))
            if new_labels == labels:
                break
            labels = new_labels[:]
            for k in range(self.num_clusters):
                cluster_points = [
                    features[i].toarray()[0]
                    for i in range(features.shape[0]) if labels[i] == k
                ]
                if cluster_points:
                    centroids[k] = np.mean(cluster_points, axis=0)
        return labels

    def train_model(self, query, generated_response=None, category=None):
        self.query_count += 1
        self.training_data["queries"].append(query)
        self.training_data["cluster_labels"].append(-1)
        if generated_response and category:
            if category not in self.training_data["generated_responses"]:
                self.training_data["generated_responses"][category] = []
            self.training_data["generated_responses"][category].append(generated_response)

        if self.query_count % self.train_interval == 0 and len(self.training_data["queries"]) >= self.num_clusters:
            try:
                features = self.vectorizer.fit_transform(self.training_data['queries'])
                self.training_data["cluster_labels"] = self.custom_cluster(features)
                self.save_training_data()
                return "Model updated successfully."
            except Exception as e:
                logger.error(f"Error updating model: {e}")
                return f"Error updating model: {e}"
        return None

    def get_query_category(self, query):
        query_lower = query.lower()
        for category, info in self.query_response_map.items():
            for keyword in info["keywords"]:
                if keyword in query_lower:
                    return category
        return None

    def generate_new_response(self, query, category):
        if category in self.query_response_map:
            existing_responses = self.query_response_map[category]["responses"]
        else:
            existing_responses = self.response_formats

        # Response templates for rephrasing
        templates = [
            "I’m here to assist with {query}. {action}",
            "Thank you for reaching out about {query}. {action}",
            "Regarding your concern with {query}, {action}",
            "I understand your request for {query}. {action}",
            "For {query}, {action}",
            "We appreciate your inquiry about {query}. {action}",
            "To address {query}, {action}"
        ]

        # Extract key phrases from existing responses
        key_phrases = []
        for resp in existing_responses:
            # Remove the {query} placeholder
            clean_resp = resp.replace("{query}", query.lower())
            # Extract meaningful phrases (e.g., after "for", "with", etc.)
            phrases = re.findall(r"(?:for|with|regarding|to|about)\s+[\w\s]+(?:\.)", clean_resp, re.IGNORECASE)
            key_phrases.extend([p.strip(".") for p in phrases])
            # Extract action-oriented clauses
            actions = re.findall(r"(?:please|kindly|you can|we recommend)\s+[\w\s]+(?:\.)", clean_resp, re.IGNORECASE)
            key_phrases.extend([a.strip(".") for a in actions])

        # Select random phrases or fallback to generic
        if key_phrases:
            action = random.choice(key_phrases)
        else:
            action = "please refer to our website for further details"

        # Combine with a random template
        new_response = random.choice(templates).format(query=query.lower(), action=action)

        # Ensure response is unique
        all_responses = existing_responses + (self.training_data["generated_responses"].get(category, []) if category in self.training_data["generated_responses"] else [])
        if new_response in all_responses:
            # Try a different combination
            action = random.choice(key_phrases) if key_phrases else "check our website for more information"
            new_response = random.choice(templates).format(query=query.lower(), action=action)

        return new_response

    def generate_initial_response(self, query, category=None):
        if query not in self.used_response_sets:
            self.used_response_sets[query] = []

        used_responses = self.used_response_sets[query]

        if category and category in self.query_response_map:
            available_formats = [
                f for f in self.query_response_map[category]["responses"]
                if f.format(query=query.lower()) not in used_responses
            ]
            if not available_formats:
                # Generate new response instead of resetting
                new_response = self.generate_new_response(query, category)
                self.query_response_map[category]["responses"].append(new_response)
                self.train_model(query, new_response, category)
                self.used_response_sets[query].append(new_response)
                return new_response
            selected_format = random.choice(available_formats)
        else:
            available_formats = [
                f for f in self.response_formats
                if f.format(query=query.lower()) not in used_responses
            ]
            if not available_formats:
                # Generate new response for generic case
                new_response = self.generate_new_response(query, None)
                self.response_formats.append(new_response)
                self.train_model(query, new_response, None)
                self.used_response_sets[query].append(new_response)
                return new_response
            selected_format = random.choice(available_formats)

        response = selected_format.format(query=query.lower())
        self.used_response_sets[query].append(response)
        return response

    def generate_response(self, query):
        update_status = self.train_model(query)
        category = self.get_query_category(query)

        if len(self.training_data["queries"]) < self.num_clusters:
            response = self.generate_initial_response(query, category)
            status = "Success"
        else:
            try:
                query_features = self.vectorizer.transform([query])
                features = self.vectorizer.transform(self.training_data["queries"])
                distances = [
                    np.linalg.norm(query_features.toarray()[0] - f.toarray()[0])
                    for f in features
                ]
                cluster_indices = [
                    i for i, label in enumerate(self.training_data["cluster_labels"])
                    if label == self.training_data["cluster_labels"][np.argmin(distances)]
                ]
                if cluster_indices:
                    if query not in self.used_response_sets:
                        self.used_response_sets[query] = []
                    available_queries = [
                        self.training_data["queries"][i] for i in cluster_indices
                        if self.training_data["queries"][i].lower() != query.lower()
                    ]
                    response = None
                    used_responses = self.used_response_sets[query]
                    if available_queries:
                        similar_query = random.choice(available_queries)
                        similar_category = self.get_query_category(similar_query)
                        if similar_category and similar_category in self.query_response_map:
                            available_formats = [
                                f for f in self.query_response_map[similar_category]["responses"]
                                if f.format(query=query.lower()) not in used_responses
                            ]
                            if not available_formats:
                                # Generate new response
                                new_response = self.generate_new_response(query, similar_category)
                                self.query_response_map[similar_category]["responses"].append(new_response)
                                self.train_model(query, new_response, similar_category)
                                self.used_response_sets[query].append(new_response)
                                response = new_response
                            else:
                                selected_format = random.choice(available_formats)
                                response = selected_format.format(query=query.lower())
                                self.used_response_sets[query].append(response)
                        else:
                            available_formats = [
                                f for f in self.response_formats
                                if f.format(query=query.lower()) not in used_responses
                            ]
                            if not available_formats:
                                # Generate new response
                                new_response = self.generate_new_response(query, None)
                                self.response_formats.append(new_response)
                                self.train_model(query, new_response, None)
                                self.used_response_sets[query].append(new_response)
                                response = new_response
                            else:
                                selected_format = random.choice(available_formats)
                                response = selected_format.format(query=query.lower())
                                self.used_response_sets[query].append(response)
                    else:
                        response = self.generate_initial_response(query, category)
                    status = "Success (clustered)"
                else:
                    response = self.generate_initial_response(query, category)
                    status = "Success"
            except Exception as e:
                logger.error(f"Model error: {e}")
                response = self.generate_initial_response(query, category)
                status = f"Model error: {e}"

        return response, status

ai = CustomAI()

@app.post("/generate-response")
async def generate_response(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response, status = ai.generate_response(query)
        return {
            "query": query,
            "response": response,
            "status": status
        }
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)