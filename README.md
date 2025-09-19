🌍 DisasterPulse

Real-Time Disaster Detection & Analysis Using AI & Social Media

🔥 Overview

DisasterPulse is an AI-powered platform designed to monitor natural disasters in real-time. By analyzing social media feeds, translating multiple languages, and extracting geolocation from images, DisasterPulse provides actionable insights on disaster type, intensity, and affected locations.

"Detect disasters faster, respond smarter."

🌟 Key Features

🌐 Multilingual Input – Detect disasters from text in any language using Google Cloud Translation API.

🖼 Image + Description Analysis – Extract GPS coordinates from images for accurate location detection.

🐦 Live Twitter Monitoring – Fetches tweets filtered by disaster keywords using Twitter API.

🤖 AI Disaster Classification – Uses Facebook BART Large MNLI for zero-shot classification.

🗺 Location Visualization – Highlights affected cities on OpenStreetMap.

📊 Confidence Scores – Shows disaster intensity as a percentage.

💾 Downloadable Data – Save fetched tweets as CSV for further analysis.

🛠 Tools & Technologies
Component	Purpose
🐍 Python	Core programming & data processing
🌐 Streamlit	Interactive web interface
🌎 OpenStreetMap	Mapping and location visualization
🤖 Transformers	Facebook BART Large MNLI for classification
🌍 Google Cloud Translation API	Multilingual text translation
🐦 Twitter API	Real-time tweet collection
🖼 EXIF + GPS	Extract coordinates from uploaded images
🗃 Pandas & NumPy	Data handling and analysis

⚠️ Note: Users must generate their own Google Cloud Translation API credentials and Twitter Bearer Token to run the app.

🔄 Workflow

Input – Upload an image with description or enter disaster-related text.

Translation – All text is translated into English for uniform analysis.

Classification – Tweets are analyzed to determine the disaster type.

Location Detection – Detect city & full address using EXIF data or text.

Disaster Prediction – Determine disaster type & intensity confidence %.

Visualization & Output – Display location, disaster type, scores, and downloadable tweet data.

🔮 Future Enhancements

Integration with Instagram, Facebook, and other social media platforms

Multi-source disaster data fusion (satellite, government alerts)

Mobile-friendly version for real-time monitoring

Advanced AI models to predict disaster severity and impact

📌 Repository & Demo

GitHub: DisasterPulse Repository

LinkedIn Demo Video: (Add your LinkedIn post link here)

💡 Why DisasterPulse?

DisasterPulse combines AI, NLP, geolocation, and social media intelligence to create a real-time disaster monitoring solution. It empowers organizations and communities to act faster, make informed decisions, and reduce the impact of natural hazards.

🤝 Contributing

We welcome contributions! Feel free to:

Open issues

Suggest enhancements

Submit pull requests

Together, we can make DisasterPulse smarter and more robust for real-time disaster detection.s
