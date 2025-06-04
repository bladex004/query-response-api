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
import os.path
import difflib
from nltk.tokenize import word_tokenize
from nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from uuid import uuid4

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__')

app)

app = FastAPI(title="AI-Driven Query Response API")

app.mount("/staticfiles", StaticFiles(directory="static"directory="static"), name="static"))

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
    return {"message": "Welcome to the AI-Driven Query Response API. Visit /docs for API documentation."})

class QueryRequest(BaseModel):
    """Pydantic model for query request payload."""
    query: str

class PersonalizedAI:
    """Personalized AI with ML, NLP, and pattern recognition."""
    
    def __init__(self):
        self.intent_map = {
            "booking": ["book", "ticket", "reserve", "purchase"],
            "cancellation": ["cancel", "cancellation", "refund"],
            "bus_status": ["bus", "late", "delay", "location", "where"],
            "payment": ["pay", "payment", "charge", "price"],
            "complaint": ["complain", "issue", "problem", "bad"],
            "feedback": ["feedback", "review", "rate"],
            "general_info": ["info", "information", "details", "how to"]
        }
        self.intent_templates = {
            "booking": [
                "I’m sorry, I’m not sure about ‘{query}’. Are you looking to book a ticket? You can do so via our website’s booking portal.",
                "Could you clarify ‘{query}’? If you’re trying to make a booking, please visit our website and select your travel details."
            ],
            "cancellation": [
                "I’m not clear on ‘{query}’. Are you asking about cancelling a ticket? You can cancel up to 15 minutes before departure via ‘Manage My Booking’."
            ],
            "bus_status": [
                "I’m unsure about ‘{query}’. Are you inquiring about a bus’s status? Please provide your booking details or check the tracking link on our website."
            ],
            "payment": [
                "I’m not certain about ‘{query}’. Are you asking about payment options? We accept credit cards, UPI, and Net Banking via our website."
            ],
            "complaint": [
                "I’m sorry, I didn’t catch ‘{query}’. Are you reporting an issue? Please provide more details or submit a complaint via our website’s support page."
            ],
            "feedback": [
                "I’m not sure about ‘{query}’. Are you providing feedback? We’d love to hear your thoughts via the survey link on our website."
            ],
            "general_info": [
                "I’m unclear on ‘{query}’. Are you seeking more information? Please check our website for details or let me know how I can assist you further."
            ],
            "unknown": [
                "I’m sorry, I couldn’t understand ‘{query}’. Could you clarify? Are you looking for help with booking, cancellation, or something else?",
                "I didn’t quite catch ‘{query}’. Could you provide more details so I can assist you better?"
            ]
        }
        self.learned_phrases = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_query(self, query):
        """Preprocess query using NLP techniques."""
        tokens = word_tokenize(query.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def detect_intent(self, query):
        """Detect intent using NLP and pattern recognition."""
        processed_query = self.preprocess_query(query)
        max_similarity = 0
        best_intent = "unknown"

        # Keyword-based matching
        for intent, keywords in self.intent_map.items():
            for keyword in keywords:
                if keyword in processed_query:
                    return intent

        # Fuzzy matching for incomplete queries
        for intent, keywords in self.intent_map.items():
            for keyword in keywords:
                similarity = difflib.SequenceMatcher(None, keyword, processed_query).ratio()
                if similarity > max_similarity and similarity > 0.7:
                    max_similarity = similarity
                    best_intent = intent

        # Check learned phrases
        for phrase, intent in self.learned_phrases.items():
            similarity = difflib.SequenceMatcher(None, phrase.lower(), processed_query).ratio()
            if similarity > max_similarity and similarity > 0.8:
                max_similarity = similarity
                best_intent = intent

        return best_intent

    def learn_phrase(self, query, intent):
        """Learn new phrases for self-training."""
        processed_query = self.preprocess_query(query)
        if processed_query not in self.learned_phrases:
            self.learned_phrases[processed_query] = intent
            logger.info(f"Learned new phrase: '{processed_query}' for intent: {intent}")

    def generate_dynamic_response(self, query, intent):
        """Generate a new response dynamically."""
        templates = [
            "I’m here to assist with {query}. {action}",
            "Thank you for asking about {query}. {action}",
            "Regarding {query}, {action}"
        ]
        actions = [
            "please visit our website for more details",
            "you can find more information on our support page",
            "kindly provide additional details for assistance"
        ]
        return random.choice(templates).format(query=query.lower(), action=random.choice(actions))

    def generate_response(self, query, used_responses):
        """Generate a response with decision-making and non-repeating logic."""
        intent = self.detect_intent(query)
        self.learn_phrase(query, intent)
        
        templates = self.intent_templates.get(intent, self.intent_templates["unknown"])
        available_templates = [t for t in templates if t.format(query=query.lower()) not in used_responses]
        
        if not available_templates:
            new_response = self.generate_dynamic_response(query, intent)
            self.intent_templates[intent].append(new_response)
            logger.info(f"Generated new response for intent {intent}: {new_response}")
            return new_response
        
        return random.choice(available_templates).format(query=query.lower())

class CustomAI:
    """AI-driven query response generator with advanced AI features."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.training_data = {
            "queries": [],
            "cluster_labels": [],
            "generated_responses": {},
            "learned_phrases": {},
            "used_responses": {}
        }
        self.data_file = os.getenv("DATA_FILE", "/tmp/training_data.json")  # Fallback to /tmp
        self.query_count = 0
        self.train_interval = 5
        self.used_response_sets = {}
        self.num_clusters = 10
        self.personalized_ai = PersonalizedAI()
        self.query_response_map = self.load_response_map()
        self.load_training_data()

    def load_response_map(self):
        """Load FlixCRM response map with ~200+ responses."""
        return {
            "greeting": {
                "keywords": ["hello", "hi", "help", "welcome"],
                "responses": [
                    "Hello and welcome to Flix! I’m here to assist you today. How may I assist you?",
                    "Greetings and welcome to Flix. How may I help you?",
                    "Welcome to Flix! My name is your assistant. How may I assist you?",
                    "Hi, Welcome to Flix, I am your assistant. How may I assist you?"
                ]
            },
            "acknowledgment": {
                "keywords": ["reaching out", "contacted", "query", "concern"],
                "responses": [
                    "I appreciate that you have reached out to us with your query.",
                    "Thank you for reaching out with your inquiry.",
                    "I am grateful that you contacted us regarding your concern.",
                    "I appreciate that you are reporting to us about this concern. I will look into it immediately."
                ]
            },
            "verification": {
                "keywords": ["verify", "booking details", "pnr", "passenger details"],
                "responses": [
                    "Sure, I'm here to help. To assist you effectively, please provide the following details: Booking number or PNR Number, Passenger's full name, Booking email address or booking phone number. Once you share this information, I'll be able to assist you promptly.",
                    "Certainly, I'm ready to assist you. To ensure I can help you effectively, please provide the following details: Booking number, Passenger's full name, Booking email address or booking phone number. Once you provide these details, I'll be able to assist you accordingly.",
                    "Could you please provide the following details to help us complete the verification process? Passenger's Full name, Booking reference number/PNR number, Booking email address, Phone number used for booking.",
                    "To proceed with the verification, may I kindly ask you to share the following information? Passenger's Full name, Booking reference number/PNR number, Booking email address, Phone number used for booking."
                ]
            },
            "checking_details": {
                "keywords": ["checking", "looking into", "moment"],
                "responses": [
                    "Thank you for sharing the details. Please allow me a moment to look for your concern.",
                    "Thank you for providing the information. Please allow me a moment to check with the details.",
                    "Thank you for sharing these details. Just a moment, please, I'm checking on the same."
                ]
            },
            "still_checking": {
                "keywords": ["still checking", "more time"],
                "responses": [
                    "I am still checking your booking/ride details. Please be with me.",
                    "I got booking however need to check the issue with your ride."
                ]
            },
            "appreciate_patience": {
                "keywords": ["patience", "thank you", "appreciate"],
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
            "change_boarding_point": {
                "keywords": ["change boarding point", "board from another location", "different boarding"],
                "responses": [
                    "May I know why you want to change the boarding point for {query}?",
                    "As per the T&C, your ticket is valid for the boarding location booked, and I regret to inform that boarding from another location isn’t possible for {query}. https://help.flixbus.com/s/article/PSSPCan-I-board-the-bus-at-a-later-stop?language=en_IN",
                    "I understand your preference for boarding the bus from your desired location. However, to ensure a smooth process, we kindly ask you to board from the designated location where the QR code can be scanned by the bus staff.",
                    "While I appreciate your desire to board the bus from a different location, please note that the process requires the QR code to be scanned by the host at the specified boarding point. We recommend using the designated location for a seamless experience.",
                    "I understand your request to board the bus from your chosen location. Unfortunately, due to our procedures, the QR code must be scanned by the host at the specified point. We encourage you to use the designated boarding location for a smooth transition.",
                    "Thank you for your understanding. Although we recognize your preferred boarding location, our process mandates that the QR code be scanned by the host at the assigned area. For a hassle-free experience, please board from the designated location.",
                    "I appreciate your interest in boarding from your preferred spot. However, to adhere to our process, the QR code must be scanned by the host at the designated boarding area. We recommend you board from this location for a smooth experience."
                ]
            },
            "boarding_point_details": {
                "keywords": ["boarding point details", "where to board"],
                "responses": [
                    "The ticket contains essential details such as the boarding points, bus route number, a link in the bottom right corner, and a GPS link that can be accessed by clicking on the boarding point for more information.",
                    "You can find important information on the ticket, including boarding points, the bus route number, a link at the bottom right corner, and a GPS link accessible by clicking on the boarding point.",
                    "The ticket provides various details, such as the boarding points, the bus route number, a link located in the bottom right corner, and a GPS link available by clicking on the boarding point for further information."
                ]
            },
            "price_difference": {
                "keywords": ["price difference", "why price changed", "fare difference"],
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
            "bus_late": {
                "keywords": ["bus late", "bus delayed", "bus running late"],
                "responses": [
                    "I regret to inform you that the bus is running late due to some operational reason and apologize for the inconvenience caused due to delay. So requesting you to please be available on the boarding point as the ride has already been departed from the previous boarding point and now going to reach your boarding point as quickly as possible.",
                    "As per the updated information the expected time of the ride to reach your boarding point is [ETA]. So please be rest assured that the ride can reach to your location anytime, and be prepared with the boarding related documents for taking a safe and pleasant ride with us. Route Number: [Number]. Additionally I’m sharing the bus host contact number for your better assistance: [Contact Number]."
                ]
            },
            "missed_bus": {
                "keywords": ["missed bus", "miss bus"],
                "responses": [
                    "We see you as a valued customer and do not wish to leave you behind. However, it’s necessary to move the bus on schedule, as we have a commitment to punctuality.",
                    "We appreciate you as a valued customer and have no intention of leaving you behind. At the time we must adhere to the bus's scheduled departure time for the sake of punctuality.",
                    "We truly value you as a customer, and it’s not in our interest to leave you behind. However, we also have to ensure the bus departs on time, as punctuality is essential.",
                    "I sincerely apologize for the inconvenience caused by missing the bus. Please help me with the booking reference number or the PNR number for verification so that I can assist you further.",
                    "I understand how frustrating this can be. I would like to inform you that the ride has departed from the designated boarding point as per the mentioned time of the ticket and all the reminder messages have been provided to you via SMS and email.",
                    "After a careful investigation I must conclude that as there are no operational issues associated with your ride, I’m unable to provide any alternative nor process with the refund. To avoid such situation I recommend reaching out to us before departure time. May I know is it possible to reach the next boarding point?"
                ]
            },
            "bus_host_info": {
                "keywords": ["bus number", "host number"],
                "responses": [
                    "I really apologize that I am unable to provide the bus number however you may identify your ride through the route number mentioned on the ticket and it gets displayed in front of the bus as well.",
                    "I regret to inform you that we don't have access to the bus driver or host's contact information. So, we always request our passengers to be at the pickup point 15 minutes prior to the departure. You may track your ride through the tracing link mentioned on the ticket."
                ]
            },
            "where_is_bus": {
                "keywords": ["where is my bus", "bus location"],
                "responses": [
                    "As checked, the ride is for [Details]. So, I would like to inform you that the ride will arrive at at the boarding point as per per the mentioned time on the ticket. I’m requesting you to please be present at the boarding point 15 minutes prior to prior to the departure time. Here I’m sharing the bus route number: [Number]. Additionally, I’m sharing the tracking link link: https://global.flixbus.com/track/order/3242800682 and the bus host number: [Number]."
                ]
            },
            "where_is_boarding_point": {
                "keywords": ["where is boarding point", "boarding point location"],
                "responses": [
                    "As checked the ticket is for [Details]. This is the address of your boarding point: [Address]. This is the Google map link for for the image of your of your boarding point: [Link]. This is the bus host contact number point: [Number]. I’m requesting you to please to be available on the boarding point."
                ]
            },
            "bus_delay": {
                "keywords": ["bus delay", "delay"],
                "responses": [
                    "I’m sincerely apologizing for the delay of the ride delay and sorry for delay the inconvenience caused to by you.",
                    "I would like to inform to you that due to some operational reason to the delay.",
                    "Here is I’m sharing the bus route number: [Number].",
                    "This is the tracking link: [Link].",
                    "Also I’m sharing the this is the bus host contact number: [Number].",
                    "Please wait any nearby shaded area any time and keep hydrating yourself."
                ]
            },
            "pax_running_late": {
                "keywords": ["running late", "suitable"],
                "responses": [
                    "Extremely sorry to inform to you that for the ride that will be departing as per its departure time scheduled.",
                    "So, requesting you to please to try",
                    "Kindly try to reach the to boarding point prior to "If you’re not going to reach reach in time you can cancel your ride up to prior to departure via prior to departure via",
                    "This is manage my booking: https://shop.flixbus.in/rebooking/ login.",
                    "This is the bus host contact number."
                ]
            },
            "ride_cancellations": {
                "keywords": ["ride cancellation", "canceled ride"],
                "responses": [
                    "I’m really sorry for this inconvenience due to "Due to ride cancellation cancellation.",
                    "A self-help link has been provided provided to you via email with with booking email id.",
                    "Please check email email or spam folder folder to book an alternative or generate refund a ticket refund within 7 days working days.",
                    "Should I proceed with your refund permission?"
                ]
            },
            "pax_no_show": {
                "keywords": ["no show", "Pax"],
                "responses": [
                    "I deeply apologize for this inconvenience.",
                    "After investigating, we find we found the bus arrived at at the designated point boarding point and other passengers.",
                    "Unfortunately we cannot process refund a refund as per policy.",
                    "https://www.flixbus.com/ terms_and_conditions.pdf, clause 15."
                ]
            },
            "changes_after_booking": {
                "keywords": ["change booking", "modify booking"],
                "responses": [
                    "After booking is done, we cannot modify ticket it from our end side.",
                    "Kindly visit our website and go to 'Manage My Booking', and fill out information in the required details.",
                    "This is the website link: https://shop.flixbus.in/rebooking."
                ]
            },
            "booking_process": {
                "keywords": ["how to book", "book process"],
                "responses": [
                    "To book, click on link provided and select 'CONTINUE' to proceed to checkout page.",
                    "Fill out: Seat, Reservation, Passengers, Contact, Payment info.",
                    "Seats: Standard seat, free, Panorama seat, Premium seat.",
                    "Note: Gender seating policy applies.",
                    "Carry ID: Aadhar, Passport, or License DL.",
                    "Luggage: 7kg hand, 20kg regular free luggage.",
                    "Payment: Credit card, UPI, Net banking.",
                    "Platform fee: Rs 5."
                ]
            },
            "manage_my_bookings": {
                "keywords": ["change date", "cancel ticket"],
                "responses": [
                    "To change date/time, cancel, or postpone ride",
                    "Use 'Manage My Booking' section.",
                    "Enter booking number, phone number number, email, click 'Retrieve Booking'.",
                    "See options to modify details.",
                    "Link: https://shop.flixbus.in/rebooking."
                ]
            },
            "complaint_feedback": {
                "keywords": ["complaint", "feedback"],
                "responses": [
                    "Thank you for your feedback!",
                    "We’re thrilled to hear you enjoyed your experience with us us.",
                    "We strive to provide excellent service always."
                ]
            },
            "driver_host_complaints": {
                "keywords": ["rude driver", "bad host"],
                "responses": [
                    "I apologize for the unpleasant experience with driver or host.",
                    "We regret their behavior was not up to standard standards.",
                    "I’ll escalate this to our team for review and action."
                ]
            },
            "bus_breakdowns": {
                "keywords": ["bus breakdown", "AC not working"],
                "responses": [
                    "I apologize for inconvenience due to bus breakdown or AC issue.",
                    "Please share booking reference number for further assist assistance.",
                    "Our team is working to resolve this issue.",
                    "As ride is ongoing, we cannot refund at this moment."
                ]
            },
            "route_detail": {
                "keywords": ["route details", "bus route"],
                "responses": [
                    "We don’t have specific route info information for the ride.",
                    "We have access to stop locations: [Stop Locations].",
                    "Tracking link: [Link]."
                ]
            },
            "change_dates": {
                "keywords": ["change date", "reschedule"],
                "responses": [
                    "You can change journey date up to 15 min minutes prior departure via prior via manage booking.",
                    "Go to: https://shop.flixbus.in/rebooking/login.",
                    "Enter Booking number, Email/Phone, click Retrieve.",
                    "Select 'change departure date'.",
                    "Note: Dynamic pricing may apply."
                ]
            },
            "route_information": {
                "keywords": ["route info", "bus path"],
                "responses": [
                    "We don’t have route info information for the ride.",
                    "We have access to stop locations.",
                    "View route: https://www.flixbus.in/track/ with booking number."
                ]
            },
            "flix_lounge": {
                "keywords": ["flix lounge", "anand vihar"],
                "responses": [
                    "Flix Lounge is not available at Anand Vihar location.",
                    "It’s an operational boarding point only.",
                    "Arrive 15 min minutes prior to departure."
                ]
            },
            "delay_under_120": {
                "keywords": ["delay under 120", "short delay"],
                "responses": [
                    "Sorry for the bus delay due to operational or traffic issues.",
                    "Delay is under 120 min minutes, no refund per T&C.",
                    "See: https://www.flixbus.in/terms, clause 15."
                ]
            },
            "delay_over_120": {
                "keywords": ["delay over 120", "long delay"],
                "responses": [
                    "I apologize for delay over 2 hours due to operational issues.",
                    "I can cancel ticket and refund if you prefer not to wait.",
                    "Proceed with refund? It’ll credit in 7 working days."
                ]
            },
            "breakdown_no_refunded": {
                "keywords": ["breakdown no refund", "AC issue"],
                "responses": [
                    "Sorry for inconvenience due to breakdown or AC issue.",
                    "As ride is ongoing, we cannot refund now.",
                    "Our team is addressing the issue."
                ]
            },
            "luggage_policies": {
                "keywords": ["luggage policy", "baggage rules"],
                "responses": [
                    "Luggage: 7kg hand, 20kg regular free luggage.",
                    "Extra 20kg luggage bookable via Manage Booking: https://shop.flixbus.in/rebooking.",
                    "Special luggage (up to 30kg) bookable.",
                    "See prices: https://www.flixbus.in/prices/baggage."
                ]
            },
            "cancel_tickets": {
                "keywords": ["cancel ticket", "cancellation"],
                "responses": [
                    "Cancel ticket up to 15 min minutes prior departure via prior Manage Booking.",
                    "Go to: https://shop.flixbus.in/rebooking/login.",
                    "Select 'Cancel trip'.",
                    "Choose cash or voucher refund (7 days).",
                    "See policy: [Link]."
                ]
            },
            "stranded_passenger": {
                "keywords": ["stranded", "left behind"],
                "responses": [
                    "Sorry to hear you were stranded.",
                    "Please share booking number, email, phone number details.",
                    "We cannot refund or arrange alternative as per policy.",
                    "Please make alternate travel arrangements."
                ]
            },
            "lost_items": {
                "keywords": ["lost item", "left something"],
                "responses": [
                    "Sorry to hear you left belongings on the bus.",
                    "Fill out Lost and Found form to recover items.",
                    "Our team will investigate and contact you."
                ]
            },
            "pet_travel": {
                "keywords": ["pet travel", "pet policy"],
                "responses": [
                    "We cannot accommodate pets on buses at this time.",
                    "See pet policy for details."
                ]
            },
            "price_discounts": {
                "keywords": ["price", "discounts"],
                "responses": [
                    "Ticket price is final, no discounts available now.",
                    "Prices are dynamic based on demand.",
                    "Book early to secure best price."
                ]
            },
            "blanket_services": {
                "keywords": ["blanket", "blanket service"],
                "responses": [
                    "We don’t provide blankets on board currently.",
                    "Carry one for comfort.",
                    "Some rides offer blankets and water."
                ]
            },
            "water_bottle_services": {
                "keywords": ["water bottle", "water service"],
                "responses": [
                    "We don’t offer water bottles on Flix buses.",
                    "Bring your own water and refreshments."
                ]
            },
            "washroom_services": {
                "keywords": ["washroom", "restroom"],
                "responses": [
                    "Flix buses don’t have washroom facilities.",
                    "Bus host will arrange comfort breaks."
                ]
            },
            "seat_changes": {
                "keywords": ["change seat", "seat change"],
                "responses": [
                    "We cannot change seats as they’re auto-assigned.",
                    "Seats are system-generated based on availability."
                ]
            },
            "shadow_bookings": {
                "keywords": ["shadow booking", "booking not found"],
                "responses": [
                    "I couldn’t find booking with provided details.",
                    "Please share: Passenger name, Email, Phone, Payment screenshot."
                ]
            },
            "no_refunded": {
                "keywords": ["no refund", "pax no show"],
                "responses": [
                    "After investigation, bus arrived at boarding point.",
                    "Other passengers boarded, so no refund possible."
                ]
            },
            "refund_processing": {
                "keywords": ["refund processing", "refund time"],
                "responses": [
                    "Ticket cancelled on [DATE].",
                    "Refund of [AMOUNT] will credit in 7 working days."
                ]
            },
            "refund_tat_crossed": {
                "keywords": ["refund not received", "refund delayed"],
                "responses": [
                    "Refund of [AMOUNT] processed on [DATE].",
                    "Check with bank or share bank statement.",
                    "Contact: https://help.flixbus.com/cancellation."
                ]
            },
            "closing": {
                "keywords": ["bye", "thank you", "end"],
                "responses": [
                    "Thank you for contacting Flix. Have a great day!",
                    "Happy to assist! Reach out for more questions.",
                    "Pleasure assisting you. Contact us again if needed.",
                    "Chat paused. Start new session via support page."
                ]
            },
            "request_rating": {
                "keywords": ["feedback", "rating", "survey"],
                "responses": [
                    "Your feedback means a lot! Share via survey link.",
                    "Glad to help! Please provide feedback via survey."
                ]
            }
        }

    def load_training_data(self):
        """Load training data from JSON file."""
        dir_path = os.path.dirname(self.data_file)
        try:
            # Ensure directory exists
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory for training data: {dir_path}")

            if not os.path.exists(self.data_file):
                logger.info(f"No training data file found at {self.data_file}, initializing empty")
                # Create empty file
                with open(self.data_file, 'w') as f:
                    json.dump(self.training_data, f, indent=2)
                return

            with open(self.data_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and "queries" in data and "cluster_labels" in data:
                    self.training_data = data.copy()
                    if "generated_responses" not in self.training_data:
                        self.training_data["generated_responses"] = {}
                    if "learned_phrases" not in self.training_data:
                        self.training_data["learned_phrases"] = {}
                    if "used_responses" not in self.training_data:
                        self.training_data["used_responses"] = {}
                    self.personalized_ai.learned_phrases = self.training_data["learned_phrases"]
                    logger.info(f"Successfully loaded training data from {self.data_file}")
                else:
                    logger.warning(f"Invalid data format in {self.data_file}, initializing empty")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.data_file}: {e}")
        except PermissionError as e:
            logger.error(f"Permission error accessing {self.data_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading data from {self.data_file}: {e}")

    def save_training_data(self):
        """Save training data to JSON file."""
        self.training_data["learned_phrases"] = self.personalized_ai.learned_phrases
        dir_path = os.path.dirname(self.data_file)
        try:
            # Ensure directory exists
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory for training data: {dir_path}")

            with open(self.data_file, mode='w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"Successfully saved training data to {self.data_file}")
        except PermissionError as e:
            logger.error(f"Permission error saving to {self.data_file}: {e}")
        except Exception as e:
            logger.error(f"Error saving data to {self.data_file}: {e}")

    def custom_cluster(self, features):
        """Perform custom k-means clustering for pattern recognition."""
        if features.shape[0] < self.num_clusters:
            return [-1] * features.shape[0]

        try:
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
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return [-1] * features.shape[0]

    def train_model(self, query, generated_response=None, category=None):
        """Train model with ML and adaptation."""
        self.query_count += 1
        self.training_data["queries"].append(query)
        self.training_data["cluster_labels"].append(-1)

        if generated_response and category:
            if category not in self.training_data["generated_responses"]:
                self.training_data["generated_responses"][category] = []
            if category not in self.training_data["used_responses"]:
                self.training_data["used_responses"][category] = set()
            self.training_data["generated_responses"][category].append(generated_response)
            self.training_data["used_responses"][category].add(generated_response)

        if self.query_count % self.train_interval == 0 and len(self.training_data["queries"]) >= self.num_clusters:
            try:
                features = self.vectorizer.fit_transform(self.training_data["queries"])
                self.training_data["cluster_labels"] = self.custom_cluster(features)
                self.save_training_data()
                logger.info("Model updated successfully")
            except Exception as e:
                logger.error(f"Error updating model: {e}")

    def get_query_category(self, query):
        """Determine category using pattern recognition."""
        query_lower = query.lower()
        for category, info in self.query_response_map.items():
            for keyword in info["keywords"]:
                if keyword in query_lower:
                    return category
        return None

    def generate_new_response(self, query, category):
        """Generate a unique response by analyzing previous data."""
        existing_responses = self.query_response_map.get(category, {}).get("responses", [])
        used_responses = self.training_data["used_responses"].get(category, set())
        all_responses = existing_responses + self.training_data["generated_responses"].get(category, [])

        # Extract key phrases and actions
        key_phrases = []
        for resp in all_responses:
            clean_resp = resp.replace("{query}", query.lower())
            phrases = re.findall(r"(?:for|with|regarding|to|about)\s+[\w\s]+(?:\.)", clean_resp, re.IGNORECASE)
            key_phrases.extend([p.strip(".") for p in phrases])
            actions = re.findall(r"(?:please|kindly|you can|we recommend)\s+[\w\s]+(?:\.)", clean_resp, re.IGNORECASE)
            key_phrases.extend([a.strip(".") for a in actions])

        # Templates for new responses
        templates = [
            "I’m here to assist with {query}. {action}",
            "Thank you for reaching out about {query}. {action}",
            "Regarding your concern with {query}, {action}",
            "I understand your query about {query}. {action}",
            "We’re here to help with {query}. {action}"
        ]

        # Try generating a unique response
        max_attempts = 10
        for _ in range(max_attempts):
            action = random.choice(key_phrases) if key_phrases else "please visit our website for further assistance"
            new_response = random.choice(templates).format(query=query.lower(), action=action)
            if new_response not in used_responses and new_response not in all_responses:
                self.training_data["used_responses"].setdefault(category, set()).add(new_response)
                self.save_training_data()
                return new_response

        # Fallback if unique response not found
        fallback_action = "check our support page for more details"
        new_response = random.choice(templates).format(query=query.lower(), action=fallback_action)
        self.training_data["used_responses"].setdefault(category, set()).add(new_response)
        self.save_training_data()
        return new_response

    def generate_initial_response(self, query, category=None):
        """Generate initial response with non-repeating logic."""
        if query not in self.used_response_sets:
            self.used_response_sets[query"] = set()

        used_responses = self.used_response_sets[query]
        persistent_used = self.training_data["used_responses"].get(category, set())

        if category and category in self.query_response_map:
            available_formats = [
                f for f in self.query_response_map[category]["responses"]
                if f.format(query=query.lower()) not in used_responses and f.format(query=query.lower()) not in persistent_used
            ]
            if not available_formats:
                response = self.generate_new_response(query, category)
                self.training_data["generated_responses"].setdefault(category, []).append(response)
                self.train_model(query, response, category)
                self.used_response_sets[query].add(response)
                self.save_training_data()
                return response
            selected_format = random.choice(available_formats)
            response = selected_format.format(query=query.lower())
            self.used_response_sets[query].add(response)
            self.training_data["used_responses"].setdefault(category, set()).add(response)
            self.save_training_data()
        else:
            response = self.personalized_ai.generate_response(query, used_responses)
            self.train_model(query, response, None)
            self.used_response_sets[query].add(response)
            self.training_data["used_responses"].setdefault("unknown", set()).add(response)
            self.save_training_data()
        return response

    def find_similar_query(self, query, features, query_features):
        """Find similar query using pattern recognition."""
        try:
            similarities = cosine_similarity(query_features, features)[0]
            cluster_indices = [
                i for i, label in enumerate(self.training_data["cluster_labels"])
                if label == self.training_data["cluster_labels"][np.argmax(similarities)]
            ]
            available_queries = [
                self.training_data["queries"][i] for i in cluster_indices
                if self.training_data["queries"][i].lower() != query.lower()
            ]
            return random.choice(available_queries) if available_queries else None
        except Exception as e:
            logger.error(f"Error finding similar query: {e}")
            return None

    def generate_response(self, query):
        """Generate a response with advanced AI features."""
        self.train_model(query)
        category = self.get_query_category(query)

        if len(self.training_data["queries"]) < self.num_clusters:
            response = self.generate_initial_response(query, category)
            return response, "Success"

        try:
            query_features = self.vectorizer.transform([query])
            features = self.vectorizer.transform(self.training_data["queries"])
            similar_query = self.find_similar_query(query, features, query_features)

            if similar_query:
                similar_category = self.get_query_category(similar_query)
                used_responses = self.used_response_sets.get(query, set())
                persistent_used = self.training_data["used_responses"].get(similar_category, set())
                if similar_category and similar_category in self.query_response_map:
                    available_formats = [
                        f for f in self.query_response_map[similar_category]["responses"]
                        if f.format(query=query.lower()) not in used_responses and f.format(query=query.lower()) not in persistent_used
                    ]
                    if not available_formats:
                        response = self.generate_new_response(query, similar_category)
                        self.training_data["generated_responses"].setdefault(similar_category, []).append(response)
                        self.train_model(query, response, similar_category)
                        self.used_response_sets[query].add(response)
                    else:
                        response = random.choice(available_formats).format(query=query.lower())
                        self.used_response_sets[query].add(response)
                        self.training_data["used_responses"].setdefault(similar_category, set()).add(response)
                        self.save_training_data()
                else:
                    response = self.personalized_ai.generate_response(query, used_responses)
                    self.train_model(query, response, None)
                    self.used_response_sets[query].add(response)
                    self.training_data["used_responses"].setdefault("unknown", set()).add(response)
                    self.save_training_data()
                return response, "Success (clustered)"
            else:
                response = self.generate_initial_response(query, category)
                return response, "Success"
        except Exception as e:
            logger.error(f"Model error: {str(e)}")
            response = self.generate_initial_response(query, category)
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
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)