
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

app = FastAPI(title="AI-Driven Query Response API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint for API welcome message."""
    return {"message": "Welcome to the AI-Driven Query Response API. Visit /docs for API documentation."}

class QueryRequest(BaseModel):
    """Pydantic model for query request payload."""
    query: str

class PersonalizedAI:
    """Personalized AI with optimized NLP and response generation."""
    
    def __init__(self):
        self.intent_map = {
            "booking": ["book", "ticket", "reserve", "purchase"],
            "cancellation": ["cancel", "cancellation", "refund"],
            "bus_status": ["bus", "late", "delay", "location", "where"],
            "payment": ["pay", "payment", "charge", "price"],
            "complaint": ["complain", "support", "problem", "issue"],
            "feedback": ["feedback", "review", "rate"],
            "general_info": ["info", "information", "details", "how"]
        }
        self.intent_templates = {
            "booking": [
                "I'm sorry, I'm not sure about '{query}'. Are you looking to book a ticket? You can do so via our website's booking portal.",
                "Could you clarify '{query}'? If you're trying to make a booking, please visit our website to select your booking details."
            ],
            "cancellation": [
                "I'm not clear on '{query}'. Are you asking about cancelling a ticket? You can cancel up to 15 minutes before departure via 'Manage My Booking'."
            ],
            "bus_status": [
                "I'm unsure about '{query}'. Are you inquiring about a bus's status? Please provide your booking details or check the tracking link on our website."
            ],
            "payment": [
                "I'm not certain about '{query}'. Are you asking about payment options? We accept credit cards, UPI, and Net Banking via our website."
            ],
            "complaint": [
                "I'm sorry, I didn't catch '{query}'. Are you reporting an issue? Please provide more details or submit a complaint via our website's support page."
            ],
            "feedback": [
                "I'm not sure about '{query}'. Are you providing feedback? We'd love to hear your thoughts via the survey link on our website."
            ],
            "general_info": [
                "I'm unclear on '{query}'. Are you seeking more information? Please check our website for details or let me know how I can assist you further."
            ],
            "unknown": [
                "I'm sorry, I couldn't understand '{query}'. Could you clarify? Are you looking for help with booking, cancellation, or something else?",
                "I didn't quite catch '{query}'. Could you provide more details so I can assist you better?"
            ]
        }
        self.learned_phrases = defaultdict(str)
        self.query_cache = {}  # Cache for preprocessed queries
        self.stop_words = set(stopwords.words('english'))

    def preprocess_query(self, query):
        """Optimized query preprocessing."""
        query_lower = query.lower()
        if query_lower in self.query_cache:
            return self.query_cache[query_lower]
        tokens = word_tokenize(query_lower)
        tokens = [token for token in tokens if token not in self.stop_words and token.isalnum()]
        processed = ' '.join(tokens)
        self.query_cache[query_lower] = processed
        return processed

    def detect_intent(self, query):
        """Detect intent with optimized keyword matching."""
        processed_query = self.preprocess_query(query)
        for intent, keywords in self.intent_map.items():
            if any(keyword in processed_query for keyword in keywords):
                return intent
        return "unknown"

    def learn_phrase(self, query, intent):
        """Learn phrases for intent mapping."""
        processed_query = self.preprocess_query(query)
        if processed_query not in self.learned_phrases:
            self.learned_phrases[processed_query] = intent
            logger.debug(f"Learned phrase: '{processed_query}' -> {intent}")

    def generate_dynamic_response(self, query, intent):
        """Generate a dynamic response."""
        templates = [
            "Regarding {query}, {action}",
            "For {query}, {action}",
            "About {query}, {action}"
        ]
        actions = [
            "visit our website for details",
            "check our support page",
            "contact support for help"
        ]
        return random.choice(templates).format(query=query.lower(), action=random.choice(actions))

    def generate_response(self, query, used_responses):
        """Generate a response with optimized logic."""
        intent = self.detect_intent(query)
        self.learn_phrase(query, intent)
        templates = self.intent_templates.get(intent, self.intent_templates["unknown"])
        available_templates = [t for t in templates if t.format(query=query.lower()) not in used_responses]
        if not available_templates:
            response = self.generate_dynamic_response(query, intent)
            logger.debug(f"Generated dynamic response for '{query}': {response}")
            return response
        response = random.choice(available_templates).format(query=query.lower())
        logger.debug(f"Selected template response for '{query}': {response}")
        return response

class CustomAI:
    """Optimized AI for fast query response."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.training_data = {
            "queries": [],
            "cluster_labels": [],
            "generated_responses": {},
            "learned_phrases": {},
            "used_responses": {},
            "intent_mappings": {}
        }
        self.data_file = os.getenv("DATA_FILE", "/tmp/training_data.json")
        self.query_count = 0
        self.train_interval = 20  # Increased to reduce clustering frequency
        self.save_interval = 10  # Save data every 10 queries
        self.used_response_sets = defaultdict(set)
        self.num_clusters = 5  # Reduced for faster clustering
        self.vectorizer_fitted = False  # Track vectorizer state
        self.personalized_ai = PersonalizedAI()
        self.query_response_map = self.load_response_map()
        self.load_training_data()

    def load_response_map(self):
        """Load predefined responses for specific query categories."""
        return {
            "greeting": {
                "keywords": ["hello", "hi", "greetings", "welcome"],
                "responses": [
                    "Hello and welcome to Flix! I'm here to assist you. How may I help you today?",
                    "Greetings and welcome to Flix. How may I assist you?",
                    "Welcome to Flix! My name is your assistant. How can I help you?",
                    "Hi, Welcome to Flix, I am your assistant. How may I assist you?"
                ]
            },
            "appreciation": {
                "keywords": ["thank you", "reaching out", "contacted", "reporting"],
                "responses": [
                    "I appreciate that you have reached out to us with your query.",
                    "Thank you for reaching out with your inquiry.",
                    "I am grateful that you contacted us regarding your concern.",
                    "I appreciate that you are reporting to us about this concern. I will look into it immediately."
                ]
            },
            "verification_details": {
                "keywords": ["booking number", "pnr", "passenger name", "verify"],
                "responses": [
                    "Sure, I'm here to help. To assist you effectively, please provide the following details: Booking number or PNR Number, Passenger's full name, Booking email address or booking phone number. Once you share this information, I'll be able to assist you promptly.",
                    "Certainly, I'm ready to assist you. To ensure I can help you effectively, please provide the following details: Booking number, Passenger's full name, Booking email address or booking phone number. Once you provide these details, I'll be able to assist you accordingly."
                ]
            },
            "processing_time": {
                "keywords": ["checking", "moment", "please wait"],
                "responses": [
                    "Thank you for sharing the details. Please allow me a moment to look for your concern.",
                    "Thank you for providing the information. Please allow me a moment to check with the details.",
                    "Thank you for sharing these details. Just a moment, please, I'm checking on the same."
                ]
            },
            "extended_processing": {
                "keywords": ["still checking", "more time"],
                "responses": [
                    "I am still checking your booking/ride details. Please be with me.",
                    "I got booking however need to check the issue with your ride."
                ]
            },
            "patience_appreciation": {
                "keywords": ["patience", "thank you for waiting"],
                "responses": [
                    "Thank you so much for your patience.",
                    "I really appreciate your patience.",
                    "Thank you so much for notifying me about the issue.",
                    "Thank you for reaching out to me about this.",
                    "I will get your issue resolved positively.",
                    "I appreciate your patience in this matter.",
                    "Your patience is appreciable."
                ]
            },
            "board_from_other_location": {
                "keywords": ["board from other", "change boarding", "different location"],
                "responses": [
                    "May I know why you want to change the boarding point and make it to other? As per the T&C, your ticket is valid for the boarding location to booked for and I regret to inform that boarding from other location isn't possible. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                    "I understand your preference for boarding the bus from your desired location. However, to ensure a smooth process, we kindly ask you to board from the designated location where the QR code can be scanned by the bus staff. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                    "While I appreciate your desire to board the bus from a different location, please note that the process requires the QR code to be scanned by the host at the specified boarding point. We recommend using the designated location for a seamless experience. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                    "I understand your request to board the bus from your chosen location. Unfortunately, due to our procedures, the QR code must be scanned by the host at the specified point. We encourage you to use the designated boarding location for a smooth transition. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                    "Thank you for your understanding. Although we recognize your preferred boarding location, our process mandates that the QR code be scanned by the host at the assigned area. For a hassle-free experience, please board from the designated location. Link: https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN"
                ]
            },
            "boarding_point_details": {
                "keywords": ["boarding point", "where to board"],
                "responses": [
                    "The ticket contains essential details such as the boarding points, bus route number, a link in the bottom right corner, and a GPS link that can be accessed by clicking on the boarding point for more information.",
                    "You can find important information on the ticket, including boarding points, the bus route number, a link at the bottom right corner, and a GPS link accessible by clicking on the boarding point.",
                    "The ticket provides various details, such as the boarding points, the bus route number, a link located in the bottom right corner, and a GPS link available by clicking on the boarding point for further information."
                ]
            },
            "price_difference": {
                "keywords": ["price difference", "why price change", "fare change"],
                "responses": [
                    "I would like to inform you that as the prices are dynamic in nature and may change on the website with time.",
                    "Please note that our prices are dynamically adjusted based on demand, availability, and other factors to ensure the best possible experience for all our passengers.",
                    "To provide you with the most accurate and fair pricing, our rates are dynamically updated. We recommend booking early to secure the best available price.",
                    "We always recommend our passengers about a price lock feature where the price shown at the time of selection is reserved for 10 minutes, allowing passengers to complete their booking at the same rate within this timeframe. Delays in booking beyond this period may result in price adjustments.",
                    "We suggest booking early and utilizing the price lock feature to secure the most favorable fare."
                ]
            },
            "pre_departure_call": {
                "keywords": ["pre departure call", "call before departure"],
                "responses": [
                    "I completely understand that you are expecting a call from the bus staff. Please note that pre-departure calls are not mandatory however, the host might call you before arriving at your departure point. We appreciate your understanding."
                ]
            },
            "bus_delay": {
                "keywords": ["bus late", "bus delay", "delayed bus"],
                "responses": [
                    "I regret to inform you that the bus is running late due to some operational reason and apologize for the inconvenience caused due to delay. So requesting you to please be available on the boarding point as the ride has already been departed from the previous boarding point and now going to reach your boarding point as quickly as possible. I appreciate your understanding.",
                    "I’m sincerely apologizing for the delay of the ride and sorry for the inconvenience caused to you. I would like to inform you that due to some operational reason the ride has been delayed and our operational team is trying to manage the delay and reach out your boarding point as soon as possible.",
                    "I regret to inform you that due to the heavy traffic the ride got stuck and now it’s get back on the track and the operational team is trying to manage that delay and reach out to your boarding point as quickly as possible. We need your patience and understanding in this matter."
                ]
            },
            "missed_bus": {
                "keywords": ["missed bus", "miss bus", "left behind"],
                "responses": [
                    "We see you as a valued customer and do not wish to leave you behind. However, it’s necessary to move the bus on schedule, as we have a commitment to punctuality. I sincerely apologize for the inconvenience caused by missing the bus.",
                    "We appreciate you as a valued customer and have no intention of leaving you behind. At the same time, we must adhere to the bus's scheduled departure time for the sake of punctuality. I understand how frustrating this can be, and I’m here to help you any further assistance you may need.",
                    "We truly value you as a customer, and it’s not in our interest to leave you behind. However, we also have to ensure the bus departs on time, as punctuality is essential. Please help me with the booking reference number or the PNR number for verification process so that I can help you with the information associated with your journey with Flix."
                ]
            },
            "bus_host_number": {
                "keywords": ["bus number", "host number", "driver number"],
                "responses": [
                    "I really apologize that I am unable to provide the bus number however you may identify your ride through the route number mentioned on the ticket and it gets displayed in front of the bus as well.",
                    "I regret to inform you that we don't have access to the bus driver or host's contact information. So, we always request our passengers to be at the pickup point 15 minutes prior to the departure, You may track your ride through the tracing link mentioned on the ticket."
                ]
            },
            "where_is_bus": {
                "keywords": ["where is bus", "bus location", "track bus"],
                "responses": [
                    "As checked, the ride is for your booked route. So, I would like to inform you that the ride will arrive at the boarding point as per the mentioned time on the ticket. I’m requesting you to please be present at the boarding point 15 minutes before the departure time of the bus, so that you can easily board the bus. Here I’m sharing the bus route number which is mentioned on the ticket just under the boarding point details and it’s also available in front of the bus from which you can easily recognize your bus. Tracking Link: https://global.flixbus.com/track/order/3242800682",
                    "I’m requesting you to please be available on the boarding point 15 minutes before the departure time of the bus, so that you can easily board the bus. Have a safe and pleasant journey ahead with Flix. Tracking Link: https://global.flixbus.com/track/order/3242800682"
                ]
            },
            "where_is_boarding_point": {
                "keywords": ["boarding point location", "where is boarding point"],
                "responses": [
                    "As checked the ticket is for your booked route. This is the address of your boarding point. This is the Google map link for the exact boarding point location. I’m requesting you to please be available on the boarding point 15 minutes before the departure time of the bus, so that you can easily board the bus. Have a safe and pleasant journey ahead with Flix."
                ]
            },
            "pax_running_late": {
                "keywords": ["running late", "late to board", "wait for me"],
                "responses": [
                    "Extremely sorry to inform you that the bus will depart from the boarding point as per the scheduled time so requesting you to please try to reach the boarding point before the mentioned time on the ticket and ensure not to miss the bus.",
                    "Unfortunately, the bus cannot wait for the delayed passengers. Our buses travel within a network and are bound to follow a timetable. Please ensure that you are at the stop at least 15 minutes before departure. If you realize that you’re not going to reach in time, you can cancel your ride up to 15 minutes before departure via manage my booking of our website. Link: https://shop.flixbus.in/rebooking/login"
                ]
            },
            "ride_cancellation": {
                "keywords": ["ride cancelled", "bus cancelled", "cancellation"],
                "responses": [
                    "I’m really sorry for the inconvenience caused to you due to the ride cancellation and I would like to inform you that due to some operational reason the ride has been cancelled and after cancellation a self-help link has been provided to you via email with the booking email id. So requesting you to please check the email inbox along with the spam folder for the same and after clicking on that link you will be able to book an alternative ride completely free of cost for any other day you want or you may cancel the trip and generate a full ticket refund for yourself which will be credited to the source account within 7 working days.",
                    "I’m really sorry for the inconvenience caused to you due to the ride cancellation. Or if you want I can help you with the full ticket refund by cancelling the ticket from our end and the refund will be credited to the source account within 7 working days excluding Saturday, Sunday and Public holidays/ Bank holidays. With your permission should I proceed with the refund?"
                ]
            },
            "no_show_refund_denial": {
                "keywords": ["no show", "missed bus refund", "didn't board"],
                "responses": [
                    "I deeply apologize for the inconvenience you're experiencing, and I understand your frustration. I'm unable to proceed with either the refund or booking an alternative ride for you at this time. As per our company policy, in such cases where the service has been provided as intended, we are unable to process a refund. Link: https://www.flixbus.in/terms-and-conditions-of-carriage, clause number 12.2.5",
                    "After thoroughly investigating the incident, we confirmed that the bus arrived at the designated boarding point, other passengers boarded successfully, and the bus departed after the scheduled time. Regrettably, under these circumstances, we are unable to issue a refund for your ticket. Link: https://www.flixbus.in/terms-and-conditions-of-carriage, clause number 12.2.5"
                ]
            },
            "booking_changes_denial": {
                "keywords": ["change booking", "modify booking"],
                "responses": [
                    "Once your ticket is booked, we are unable to modify it from our side. Kindly visit our website, go to ‘Manage My Booking,’ and fill in the required information to make changes. Link: https://shop.flixbus.in/rebooking/login",
                    "We cannot alter your booking after it has been confirmed. Please go to our website, select ‘Manage My Booking,’ and provide the needed details to make any changes. Link: https://shop.flixbus.in/rebooking/login"
                ]
            },
            "booking_process": {
                "keywords": ["how to book", "booking process", "book ticket"],
                "responses": [
                    "To book a ticket, please visit our website and click on the booking link. Proceed to the checkout page by selecting 'CONTINUE'. Fill out the necessary details: Seat Reservation, Passengers, Contact Information, Payment. Available seat types include Standard free seats, Panorama seats, and Premium seats. Note our gender seating policy ensures female travelers are not seated next to male travelers unless part of the same booking. Carry a valid ID (Aadhar, Passport, or Driving License). Luggage policy allows 7kg hand luggage and 20kg regular luggage free, with additional luggage bookable via Manage My Booking. Payment methods include Credit cards, UPI, and Net Banking. A Rs 5 platform fee applies."
                ]
            },
            "manage_booking_changes": {
                "keywords": ["change date", "change time", "cancel ticket"],
                "responses": [
                    "If you wish to change the date or time of your ride, cancel, or postpone it, you can easily make these adjustments through the ‘Manage My Booking’ section. Simply enter your booking number and phone number, click on ‘Retrieve Booking,’ and you will see the options to modify your details. Link: https://shop.flixbus.in/rebooking/login",
                    "If you need to reschedule, cancel, or postpone your ride, you can manage these changes through the ‘Manage My Booking’ portal. Enter your booking number and phone number, then click ‘Retrieve Booking’ to find the options for updating your ride details. Link: https://shop.flixbus.in/rebooking/login"
                ]
            },
            "complaint_feedback": {
                "keywords": ["complain", "feedback", "review"],
                "responses": [
                    "Thank you so much! We're thrilled to hear that you enjoyed your experience with us. We strive to provide excellent service, and it's always wonderful to receive positive feedback."
                ]
            },
            "rude_behavior": {
                "keywords": ["rude driver", "rude host", "bad behavior"],
                "responses": [
                    "I sincerely apologize for the unpleasant experience you had with the driver and the bus host. We deeply regret that their behavior was not up to the standards you expect and deserve. Please be assured that I will escalate this matter to the relevant team for a thorough review and appropriate action. Your feedback is very important to us, and we take such concerns seriously to ensure this doesn’t happen again."
                ]
            },
            "breakdown_refund": {
                "keywords": ["bus breakdown", "ac not working", "refund breakdown"],
                "responses": [
                    "Thank you for reaching out, and I sincerely apologize for the inconvenience you've experienced due to the breakdown of the bus. To assist you further and ensure we handle your request appropriately, could you please provide your booking reference number or PNR number, along with the email address or phone number associated with your booking? We are actively working to resolve this and will keep you updated."
                ]
            },
            "route_details": {
                "keywords": ["route details", "bus route"],
                "responses": [
                    "I regret to inform you that we don’t have the specific route information about the ride however we have the access for the stop locations associated with your journey with FlixB. These are the stop locations associated with your existing booking. Is there anything else I can help you with?"
                ]
            },
            "change_date": {
                "keywords": ["change date", "reschedule date"],
                "responses": [
                    "Yes sure, you can change the date of your journey up to 15 minutes before departure time of the bus via manage my booking section on our website. Link: https://shop.flixbus.in/rebooking/login. After clicking on the above link you can see the option for Booking number and Email or Phone number, then you have to fill those required details and click on the retrieve booking. After that you will be able to change the date of your journey. Please note that the prices are dynamic in nature, and any fare difference will be displayed during the rescheduling process."
                ]
            },
            "route_information": {
                "keywords": ["route information", "bus route info"],
                "responses": [
                    "I regret to inform you that we don’t have the route information of the ride however we have the access for the stop location associate with that booking. If you have already booked the ticket and want to know the route information of your ride you may click on the link provided below: https://www.flixbus.in/track/"
                ]
            },
            "flix_lounge": {
                "keywords": ["flix lounge", "anand vihar lounge"],
                "responses": [
                    "Thank you for reaching out to us. We apologize for any confusion, but please note that the Flix Lounge facility is not available at the Anand Vihar location. It serves as an operational point for boarding, and only official work takes place there. We suggest waiting at the boarding point for your bus."
                ]
            },
            "bus_delay_less_120": {
                "keywords": ["bus delay less than 120", "short delay"],
                "responses": [
                    "I’m really sorry for the delay of the bus. I understand this can be frustrating, and I sincerely apologize for the inconvenience caused. As checked the bus was delayed due to some operational reasons and traffic issues. I have checked that current status, and while the bus is delayed, it is not delayed by 120 minutes or more from your boarding point. According to our T&C, we can only offer a refund if the bus is delayed by more than 120 minutes from your boarding time. Link: https://www.flixbus.in/terms-and-conditions-of-carriage"
                ]
            },
            "bus_delay_over_120": {
                "keywords": ["bus delay over 120", "long delay"],
                "responses": [
                    "I sincerely apologize for the delay. I understand how frustrating this can be and I regret the inconvenience caused. Upon investigation, I can confirm that the bus is delayed by more than 2 hours from your boarding point due to operational reasons. If you prefer not to wait, I can proceed with cancelling your ticket and initiate a full refund for you. Would you like me to go ahead with that?"
                ]
            },
            "bus_breakdown_ac": {
                "keywords": ["ac not working", "bus breakdown ac"],
                "responses": [
                    "I sincerely apologize for the inconvenience caused due to the bus breakdown and as the AC not working. I understand how uncomfortable this must be for you, and I'm truly sorry. To assist you further, could you please share your booking reference number along with the email address or phone number used during the booking? I’ve already highlighted this issue to our team, and they are working on resolving it as soon as possible."
                ]
            },
            "luggage_policy": {
                "keywords": ["luggage policy", "baggage rules"],
                "responses": [
                    "Thank you for reaching out regarding our luggage policy. I’m happy to inform you that you are allowed to bring 7kg of hand luggage and 20kg of regular luggage completely free of charge. Additionally, you may bring one extra luggage item of 20kg per passenger. You can book additional luggage via Manage My Booking: https://shop.flixbus.in/rebooking/login. For more details, please visit: https://www.flixbus.in/service/luggage."
                ]
            },
            "cancel_ticket": {
                "keywords": ["cancel ticket", "ticket cancellation"],
                "responses": [
                    "I would like to inform you that you may cancel your ticket from your end up to 15 minutes before the departure time of the bus. You can cancel it through our website via Manage My Booking. Link: https://shop.flixbus.in/rebooking/login. After clicking on this link you can see both the option for booking number and email and phone number, then you can fill with the required details and click on the \"Retrieve Booking\" option. Then you will be able to cancel the ticket and choose between a cash refund or a voucher."
                ]
            },
            "stranded_passenger": {
                "keywords": ["stranded", "left behind"],
                "responses": [
                    "We’re very sorry to hear about your situation and understand how frustrating this must be. Could you please share your booking reference number, registered email, and phone number so we can look into this for you? Unfortunately, we are unable to offer a refund or arrange an alternative ride in this situation, as per our company policy."
                ]
            },
            "lost_item": {
                "keywords": ["lost item", "left something", "lost and found"],
                "responses": [
                    "We’re very sorry to hear that your belongings were left on the bus. We understand how important this is to you. To assist you in recovering your items, may I kindly request you to fill out our Lost and Found form? Our team will investigate the matter and do their best to locate your belongings."
                ]
            },
            "travel_with_pet": {
                "keywords": ["travel with pet", "pet policy"],
                "responses": [
                    "Thank you for your inquiry regarding traveling with pets on Flix. Unfortunately, at this time, we are unable to accommodate pets on our buses. This policy is in place to ensure a safe and pleasant experience for everyone on board. For more details, please refer to our official pet policy."
                ]
            },
            "prices_discounts": {
                "keywords": ["price", "discount", "offer"],
                "responses": [
                    "The price shown on your ticket is the final price. You do not need to pay any further amount.",
                    "I apologize, but currently, there are no offers or discounts available on our website. However, rest assured that our prices are already set to provide the most convenient options.",
                    "Please note that our prices are dynamically adjusted based on demand, availability, and other factors to ensure the best possible experience for all our passengers. We recommend booking early to secure the best available price."
                ]
            },
            "blanket_service": {
                "keywords": ["blanket", "blanket service"],
                "responses": [
                    "I regret to inform you but as of now we are not providing Blankets on board however we recommend our customers to carry one along with them for their own comfort and warmth. You may refer to this link you will get to know what kind of services Flix provide.",
                    "We're pleased to inform you that blankets and water bottles have been provided for your convenience on all rides."
                ]
            },
            "water_bottle_service": {
                "keywords": ["water", "water bottle"],
                "responses": [
                    "We regret to inform you that, as of now, we do not offer water bottle services on our Flix buses. We recommend that passengers bring their own water bottles and any other refreshments they might need for their journey."
                ]
            },
            "washroom_service": {
                "keywords": ["washroom", "restroom", "toilet"],
                "responses": [
                    "I regret to inform you that as of now Flix not provided the washroom facilities on the bus. However, the bus host will take care of the comfort breaks while taking your journey."
                ]
            },
            "seat_changes": {
                "keywords": ["change seat", "seat change"],
                "responses": [
                    "We apologize, but we are unable to change your seat as it is automatically assigned and based on availability.",
                    "Unfortunately, we cannot change your seat because seats are auto-assigned and system-generated based on current availability."
                ]
            },
            "shadow_booking": {
                "keywords": ["shadow booking", "payment not found", "booking not found"],
                "responses": [
                    "It's sad to hear about the inconvenience you're experiencing. To assist you further, could you please help me with the following details? 1) Passenger's full name 2) The email ID used for booking 3) The phone number associated with the booking 4) A screenshot of the payment transaction. This information will help us locate your booking and ensure everything is sorted out promptly.",
                    "I apologize for the inconvenience, but I couldn't locate any booking matching the details provided by you.",
                    "I regret to inform you that, I am unable to locate any booking with the provided details."
                ]
            },
            "no_refund_statement": {
                "keywords": ["no refund", "refund denial"],
                "responses": [
                    "After thoroughly investigating the incident, we found that the bus arrived at the designated boarding point and other passengers successfully boarded the bus. Unfortunately, due to these circumstances, we are unable to process a refund for your ticket."
                ]
            },
            "refund_processing": {
                "keywords": ["refund status", "refund processing"],
                "responses": [
                    "Your ticket has been cancelled as of (DATE). Please note that it will take up to 7 working days for the amount of (AMOUNT) to be credited back to your account. Don’t worry, your funds are secure and will be refunded within the maximum time frame.",
                    "We would like to inform you that your ticket has been cancelled on (DATE). The refund amount of (AMOUNT) will be processed and should appear in your account within 7 working days. Please be assured that your money is safe and will be returned within this period."
                ]
            },
            "refund_tat_crossed": {
                "keywords": ["refund not received", "late refund"],
                "responses": [
                    "We would like to inform you that the refund has been initiated from our end on [DATE] for the amount of [XXXX]. Please check with your bank regarding the status of this refund. If you do not receive the amount, kindly share your bank statement up to the current date for further assistance."
                ]
            },
            "closing_statement": {
                "keywords": ["goodbye", "bye", "thanks", "done"],
                "responses": [
                    "Thank you for contacting Flix. Have a great day!",
                    "I’m happy to have assisted you with your inquiry! If you have any other questions or need further assistance, please feel free to reach out. Have a wonderful day!",
                    "It was a pleasure assisting you today. If you need further assistance or have any more questions, don't hesitate to contact us again. Have a wonderful day!"
                ]
            },
            "request_feedback": {
                "keywords": ["feedback", "rate conversation", "survey"],
                "responses": [
                    "Looking forward for your valuable feedback towards my response, the link or the option will be there right after the chat ends.",
                    "We appreciate your feedback and look forward to hearing from you. You’ll find the link or option available once our chat concludes.",
                    "Your feedback towards my response is important for me! The link or option will be provided immediately after our conversation ends."
                ]
            }
        }

    def load_training_data(self):
        """Load training data with robust error handling."""
        try:
            if not os.path.exists(self.data_file):
                logger.info(f"Initializing empty training data file at {self.data_file}")
                with open(self.data_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_data, f, indent=4)
                return
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "queries" in data:
                    self.training_data.update(data)
                    self.personalized_ai.learned_phrases = self.training_data["learned_phrases"]
                    logger.info(f"Loaded training data from {self.data_file}")
                else:
                    logger.warning(f"Invalid data format in {self.data_file}, resetting to empty")
                    self.save_training_data()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {self.data_file}: {str(e)}, resetting to empty")
            self.save_training_data()
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}, resetting to empty")
            self.save_training_data()

    def save_training_data(self):
        """Save training data with batching."""
        self.training_data["learned_phrases"] = self.personalized_ai.learned_phrases
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=4)
            logger.debug(f"Saved training data to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving training data: {str(e)}")

    def custom_cluster(self, features):
        """Simplified clustering for performance."""
        if features.shape[0] < self.num_clusters:
            return [-1] * features.shape[0]
        try:
            indices = random.sample(range(features.shape[0]), self.num_clusters)
            centroids = features[indices].toarray()
            labels = np.zeros(features.shape[0], dtype=int)
            for _ in range(10):
                distances = np.array([np.sum((features.toarray() - c) ** 2, axis=1) for c in centroids]).T
                new_labels = np.argmin(distances, axis=1)
                if np.array_equal(labels, new_labels):
                    break
                labels = new_labels
                for k in range(self.num_clusters):
                    cluster_points = features[labels == k].toarray()
                    if cluster_points.shape[0] > 0:
                        centroids[k] = np.mean(cluster_points, axis=0)
            return labels.tolist()
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            return [-1] * features.shape[0]

    def ensure_vectorizer_fitted(self):
        """Ensure the vectorizer is fitted if training data exists."""
        if not self.vectorizer_fitted and self.training_data["queries"]:
            try:
                self.vectorizer.fit(self.training_data["queries"])
                self.vectorizer_fitted = True
                logger.debug("Fitted TF-IDF vectorizer")
            except Exception as e:
                logger.error(f"Failed to fit vectorizer: {str(e)}")

    def train_model(self, query, generated_response=None, response=None, category=None, intent=None):
        """Train the model with optimized updates."""
        start_time = time.time()
        self.query_count += 1
        self.training_data["queries"].append(query)
        self.training_data["cluster_labels"].append(-1)
        if intent:
            self.training_data["intent_mappings"][query.lower()] = intent
        if generated_response and category:
            self.training_data["generated_responses"].setdefault(category, []).append(generated_response)
            self.training_data["used_responses"].setdefault(category, set()).add(generated_response)
        if self.query_count % self.train_interval == 0 and len(self.training_data["queries"]) >= self.num_clusters:
            features = self.vectorizer.fit_transform(self.training_data["queries"])
            self.vectorizer_fitted = True
            self.training_data["cluster_labels"] = self.custom_cluster(features)
            self.save_training_data()
            logger.info(f"Model trained in {time.time() - start_time:.3f} seconds")
        elif self.query_count % self.save_interval == 0:
            self.save_training_data()

    def get_query_category(self, query):
        """Fast category detection for queries."""
        query_lower = query.lower()
        for category, info in self.query_response_map.items():
            if any(keyword.lower() in query_lower for keyword in info["keywords"]):
                return category
        return None

    def generate_new_response(self, query, category):
        """Generate a unique response for a query."""
        start_time = time.time()
        used_responses = self.training_data["used_responses"].get(category, set())
        templates = [
            "For {query}, {action}",
            "Regarding {query}, {action}",
            "About {query}, {action}"
        ]
        actions = [
            "please check our website for more details",
            "visit our support page for assistance",
            "contact our team for further assistance",
            "refer to our FAQ for quick answers"
        ]
        max_attempts = 5
        for _ in range(max_attempts):
            action = random.choice(actions)
            new_response = random.choice(templates).format(query=query.lower(), action=action)
            if new_response not in used_responses:
                self.training_data["used_responses"].setdefault(category, set()).add(new_response)
                self.training_data["generated_responses"].setdefault(category, []).append(new_response)
                logger.debug(f"Generated new response in {time.time() - start_time:.3f} seconds")
                return new_response
        new_response = random.choice(templates).format(query=query.lower(), action=random.choice(actions))
        logger.debug(f"Fallback response in {time.time() - start_time:.3f} seconds")
        return new_response

    def generate_initial_response(self, query, category=None):
        """Generate an initial response for a query."""
        start_time = time.time()
        query_key = query.lower()
        used_responses = self.used_response_sets[query_key]
        persistent_used = self.training_data["used_responses"].get(category, set())
        intent = self.personalized_ai.detect_intent(query)
        self.personalized_ai.learn_phrase(query, intent)
        if category and category in self.query_response_map:
            available_responses = [
                r for r in self.query_response_map[category]["responses"]
                if r not in used_responses and r not in persistent_used
            ]
            if not available_responses:
                response = self.generate_new_response(query, category)
                self.train_model(query, response, category, intent)
                self.used_response_sets[query_key].add(response)
                logger.info(f"Generated new response for '{query}' in {time.time() - start_time:.3f} seconds")
                return response
            response = random.choice(available_responses)
            self.used_response_sets[query_key].add(response)
            self.training_data["used_responses"].setdefault(category, set()).add(response)
            self.train_model(query, response, category, intent)
        else:
            response = self.personalized_ai.generate_response(query, used_responses)
            self.train_model(query, response, None, intent)
            self.used_response_sets[query_key].add(response)
        logger.info(f"Initial response for '{query}' in {time.time() - start_time:.3f} seconds")
        return response

    def find_similar_query(self, query, features, query_features):
        """Find a similar query using cosine similarity."""
        try:
            similarities = cosine_similarity(query_features, features)[0]
            max_similarity_idx = np.argmax(similarities)
            if similarities[max_similarity_idx] > 0.7:
                return self.training_data["queries"][max_similarity_idx]
            return None
        except Exception as e:
            logger.error(f"Error finding similar query: {str(e)}")
            return None

    def generate_response(self, query):
        """Generate a response with optimized flow."""
        start_time = time.time()
        intent = self.personalized_ai.detect_intent(query)
        self.train_model(query, intent=intent)
        category = self.get_query_category(query)
        if len(self.training_data["queries"]) < self.num_clusters:
            response = self.generate_initial_response(query, category)
            logger.info(f"Response for '{query}' generated in {time.time() - start_time:.3f} seconds")
            return response, "Success"
        self.ensure_vectorizer_fitted()
        if not self.vectorizer_fitted:
            response = self.generate_initial_response(query, category)
            logger.info(f"Response for '{query}' generated in {time.time() - start_time:.3f} seconds")
            return response, "Success"
        query_features = self.vectorizer.transform([query])
        features = self.vectorizer.transform(self.training_data["queries"])
        similar_query = self.find_similar_query(query, features, query_features)
        if similar_query:
            similar_category = self.get_query_category(similar_query)
            used_responses = self.used_response_sets[query.lower()]
            persistent_used = self.training_data["used_responses"].get(similar_category, set())
            if similar_category and similar_category in self.query_response_map:
                available_responses = [
                    r for r in self.query_response_map[similar_category]["responses"]
                    if r not in used_responses and r not in persistent_used
                ]
                if not available_responses:
                    response = self.generate_new_response(query, similar_category)
                    self.train_model(query, response, similar_category, intent)
                    self.used_response_sets[query.lower()].add(response)
                else:
                    response = random.choice(available_responses)
                    self.used_response_sets[query.lower()].add(response)
                    self.training_data["used_responses"].setdefault(similar_category, set()).add(response)
                    self.train_model(query, response, similar_category, intent)
            else:
                response = self.personalized_ai.generate_response(query, used_responses)
                self.train_model(query, response, None, intent)
                self.used_response_sets[query.lower()].add(response)
        else:
            response = self.generate_initial_response(query, category)
        logger.info(f"Response for '{query}' generated in {time.time() - start_time:.3f} seconds")
        return response, "Success"

ai = CustomAI()

@app.post("/generate-response")
async def generate_response(request: QueryRequest):
    """Generate a response for a user query."""
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
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)