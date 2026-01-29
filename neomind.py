"""
NeoMind - AI-Based Early Detection of Neurodevelopmental Disorders
Complete Prototype - Run with: python neomind_prototype.py
Then open: http://localhost:5000
"""

import os
import numpy as np
import pandas as pd
import json
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# ========== Flask Web Framework ==========
from flask import Flask, render_template, request, jsonify, send_file, session
import threading

# ========== Computer Vision & Audio Processing ==========
import cv2
import librosa
import soundfile as sf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ========== Machine Learning ==========
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import random
from datetime import datetime

# ========== Create Flask App ==========
app = Flask(__name__)
app.secret_key = 'neomind-prototype-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== Simulated AI Models ==========
class NeoMindAIAnalyzer:
    """Simulated AI analyzer for neurodevelopmental markers"""
    
    def __init__(self):
        self.disorders = ['ASD', 'ADHD', 'Down Syndrome', 'Developmental Delay']
        self.markers = {
            'eye_contact': {'normal': 0.8, 'threshold': 0.4},
            'motor_movements': {'normal': 0.7, 'threshold': 0.3},
            'vocal_patterns': {'normal': 0.75, 'threshold': 0.35},
            'social_smiling': {'normal': 0.85, 'threshold': 0.45},
            'response_to_name': {'normal': 0.9, 'threshold': 0.5}
        }
        self.load_simulated_models()
    
    def load_simulated_models(self):
        """Simulate loading pre-trained models"""
        print("Loading simulated AI models...")
        self.eye_tracking_model = "Loaded: Eye Gaze Analyzer v2.1"
        self.motor_analysis_model = "Loaded: Movement Pattern Detector v1.5"
        self.audio_analysis_model = "Loaded: Vocal Pattern Classifier v3.0"
        print("AI Models loaded successfully!")
    
    def analyze_video(self, video_path):
        """Analyze baby movements and facial expressions"""
        print(f"Analyzing video: {video_path}")
        
        # Simulate video analysis
        results = {
            'eye_contact_score': random.uniform(0.2, 0.95),
            'gaze_following': random.uniform(0.3, 0.9),
            'facial_expressivity': random.uniform(0.4, 0.95),
            'motor_coordination': random.uniform(0.25, 0.9),
            'limb_symmetry': random.uniform(0.6, 0.98),
            'movement_smoothness': random.uniform(0.5, 0.95)
        }
        
        # Generate risk indicators
        risks = []
        if results['eye_contact_score'] < 0.4:
            risks.append("Reduced eye contact detected")
        if results['motor_coordination'] < 0.35:
            risks.append("Motor coordination concerns")
        if results['facial_expressivity'] < 0.45:
            risks.append("Limited facial expressions")
        
        return {
            'metrics': results,
            'risk_indicators': risks,
            'summary': f"Video analysis complete: {len(risks)} potential markers found"
        }
    
    def analyze_audio(self, audio_path):
        """Analyze crying and vocalization patterns"""
        print(f"Analyzing audio: {audio_path}")
        
        try:
            # Try to load actual audio if provided
            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path, duration=10)
                duration = len(y) / sr
                
                # Extract some real features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfccs, axis=1)
                
                # Simulate more detailed analysis
                pitch_mean = np.mean(librosa.yin(y, fmin=80, fmax=400))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            else:
                # Fallback to simulated data
                duration = random.uniform(3, 15)
                mfcc_mean = np.random.randn(13)
                pitch_mean = random.uniform(150, 350)
                spectral_centroid = random.uniform(1000, 3000)
        except:
            # Simulated analysis
            duration = random.uniform(3, 15)
            mfcc_mean = np.random.randn(13)
            pitch_mean = random.uniform(150, 350)
            spectral_centroid = random.uniform(1000, 3000)
        
        # Analyze patterns
        results = {
            'duration_seconds': duration,
            'pitch_variability': random.uniform(0.1, 0.9),
            'cry_rhythm_consistency': random.uniform(0.3, 0.95),
            'vocalization_complexity': random.uniform(0.2, 0.9),
            'spectral_features': {
                'pitch_mean': float(pitch_mean),
                'spectral_centroid': float(spectral_centroid),
                'mfcc_variance': float(np.var(mfcc_mean))
            }
        }
        
        # Risk indicators
        risks = []
        if results['pitch_variability'] < 0.25:
            risks.append("Monotonous vocal patterns")
        if results['cry_rhythm_consistency'] < 0.3:
            risks.append("Atypical cry rhythm")
        if results['vocalization_complexity'] < 0.35:
            risks.append("Limited vocal complexity")
        
        return {
            'metrics': results,
            'risk_indicators': risks,
            'summary': f"Audio analysis complete: {len(risks)} vocal pattern concerns"
        }
    
    def analyze_health_data(self, health_data):
        """Analyze health records and genetic markers"""
        print("Analyzing health data...")
        
        risk_factors = []
        protective_factors = []
        
        # Analyze birth metrics
        if 'birth_weight' in health_data:
            bw = health_data['birth_weight']
            if bw < 2.5:
                risk_factors.append(f"Low birth weight: {bw} kg")
            elif bw > 4.0:
                risk_factors.append(f"High birth weight: {bw} kg")
            else:
                protective_factors.append("Normal birth weight")
        
        # Analyze Apgar scores
        if 'apgar_1min' in health_data and 'apgar_5min' in health_data:
            apgar1 = health_data['apgar_1min']
            apgar5 = health_data['apgar_5min']
            
            if apgar1 < 7:
                risk_factors.append(f"Low 1-min Apgar: {apgar1}")
            if apgar5 < 7:
                risk_factors.append(f"Low 5-min Apgar: {apgar5}")
            if apgar5 - apgar1 >= 2:
                protective_factors.append("Good Apgar score improvement")
        
        # Family history
        if 'family_history' in health_data:
            fh = health_data['family_history'].lower()
            disorders = ['autism', 'adhd', 'down', 'developmental']
            for disorder in disorders:
                if disorder in fh:
                    risk_factors.append(f"Family history of {disorder}")
        
        return {
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'total_risk_factors': len(risk_factors)
        }
    
    def calculate_risk_scores(self, video_analysis, audio_analysis, health_analysis):
        """Calculate comprehensive risk scores"""
        
        # Base scores from video analysis
        video_risk = len(video_analysis['risk_indicators']) / 5
        
        # Base scores from audio analysis
        audio_risk = len(audio_analysis['risk_indicators']) / 3
        
        # Health data risk
        health_risk = min(health_analysis['total_risk_factors'] / 5, 1.0)
        
        # Calculate disorder-specific risks
        risk_scores = {}
        explanations = {}
        
        # ASD Risk
        asd_base = (video_risk * 0.5 + audio_risk * 0.3 + health_risk * 0.2)
        asd_score = min(asd_base * (1 + random.uniform(-0.1, 0.1)), 0.95)
        risk_scores['ASD'] = round(asd_score, 3)
        explanations['ASD'] = f"Based on {len(video_analysis['risk_indicators'])} behavioral markers"
        
        # ADHD Risk
        adhd_base = (video_risk * 0.3 + audio_risk * 0.4 + health_risk * 0.3)
        adhd_score = min(adhd_base * (1 + random.uniform(-0.1, 0.1)), 0.9)
        risk_scores['ADHD'] = round(adhd_score, 3)
        explanations['ADHD'] = "Primarily from activity and attention patterns"
        
        # Down Syndrome Risk (more health-data dependent)
        down_base = (video_risk * 0.2 + audio_risk * 0.2 + health_risk * 0.6)
        down_score = min(down_base * (1 + random.uniform(-0.05, 0.15)), 0.85)
        risk_scores['Down Syndrome'] = round(down_score, 3)
        explanations['Down Syndrome'] = "Health markers and physical features analysis"
        
        # General Developmental Delay
        dd_base = (video_risk * 0.4 + audio_risk * 0.4 + health_risk * 0.2)
        dd_score = min(dd_base * (1 + random.uniform(-0.1, 0.1)), 0.92)
        risk_scores['Developmental Delay'] = round(dd_score, 3)
        explanations['Developmental Delay'] = "Overall developmental progress assessment"
        
        # Determine overall risk level
        max_risk = max(risk_scores.values())
        if max_risk > 0.7:
            overall_risk = "High"
        elif max_risk > 0.4:
            overall_risk = "Medium"
        else:
            overall_risk = "Low"
        
        return {
            'risk_scores': risk_scores,
            'overall_risk': overall_risk,
            'explanations': explanations,
            'components': {
                'video_risk': round(video_risk, 3),
                'audio_risk': round(audio_risk, 3),
                'health_risk': round(health_risk, 3)
            }
        }
    
    def generate_recommendations(self, risk_scores):
        """Generate personalized recommendations based on risk assessment"""
        
        recommendations = []
        
        for disorder, score in risk_scores['risk_scores'].items():
            if score > 0.6:
                recommendations.append({
                    'disorder': disorder,
                    'priority': 'High',
                    'actions': [
                        f"Schedule consultation with pediatric neurologist",
                        f"Begin early intervention assessment",
                        f"Monitor specific developmental milestones weekly",
                        f"Consider genetic counseling if score > 0.75"
                    ]
                })
            elif score > 0.3:
                recommendations.append({
                    'disorder': disorder,
                    'priority': 'Medium',
                    'actions': [
                        f"Discuss findings with pediatrician",
                        f"Track development with milestone checklist",
                        f"Consider developmental screening at next visit",
                        f"Engage in targeted play activities"
                    ]
                })
        
        # General recommendations
        general_recs = [
            "Maintain regular well-baby checkups",
            "Document developmental milestones",
            "Engage in face-to-face interaction daily",
            "Monitor response to name and social smiling"
        ]
        
        # Add therapy suggestions if high risk
        if risk_scores['overall_risk'] == 'High':
            general_recs.extend([
                "Consider early intervention program referral",
                "Explore speech and occupational therapy options",
                "Join parent support groups"
            ])
        
        return {
            'disorder_specific': recommendations,
            'general': general_recs,
            'follow_up_timeline': '2-4 weeks for high risk, 3-6 months for medium risk'
        }

# ========== Initialize AI Analyzer ==========
ai_analyzer = NeoMindAIAnalyzer()

# ========== Web Routes ==========
@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Data upload page"""
    return render_template('upload.html')

@app.route('/analyze')
def analyze_page():
    """Analysis results page"""
    if 'analysis_results' not in session:
        # Generate demo results
        demo_data()
    return render_template('analyze.html')

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """API endpoint to upload and analyze data"""
    try:
        # Get form data
        baby_name = request.form.get('baby_name', 'Baby Demo')
        birth_date = request.form.get('birth_date', '2024-01-01')
        birth_weight = float(request.form.get('birth_weight', 3.2))
        apgar_1min = int(request.form.get('apgar_1min', 8))
        apgar_5min = int(request.form.get('apgar_5min', 9))
        family_history = request.form.get('family_history', 'None')
        
        # Store in session
        session['baby_info'] = {
            'name': baby_name,
            'birth_date': birth_date,
            'birth_weight': birth_weight,
            'apgar_1min': apgar_1min,
            'apgar_5min': apgar_5min,
            'family_history': family_history
        }
        
        # Save uploaded files
        video_file = request.files.get('video')
        audio_file = request.files.get('audio')
        
        video_path = None
        audio_path = None
        
        if video_file and video_file.filename:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
            video_file.save(video_path)
        
        if audio_file and audio_file.filename:
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_audio.wav')
            audio_file.save(audio_path)
        
        # If no files uploaded, use demo paths
        if not video_path or not os.path.exists(video_path):
            video_path = 'demo_video.mp4'  # Simulated
        
        if not audio_path or not os.path.exists(audio_path):
            audio_path = 'demo_audio.wav'  # Simulated
        
        # Run analysis
        health_data = {
            'birth_weight': birth_weight,
            'apgar_1min': apgar_1min,
            'apgar_5min': apgar_5min,
            'family_history': family_history
        }
        
        print("Starting comprehensive analysis...")
        video_analysis = ai_analyzer.analyze_video(video_path)
        audio_analysis = ai_analyzer.analyze_audio(audio_path)
        health_analysis = ai_analyzer.analyze_health_data(health_data)
        risk_scores = ai_analyzer.calculate_risk_scores(video_analysis, audio_analysis, health_analysis)
        recommendations = ai_analyzer.generate_recommendations(risk_scores)
        
        # Prepare results
        results = {
            'baby_info': session['baby_info'],
            'video_analysis': video_analysis,
            'audio_analysis': audio_analysis,
            'health_analysis': health_analysis,
            'risk_scores': risk_scores,
            'recommendations': recommendations,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_id': f"NM{random.randint(10000, 99999)}"
        }
        
        # Store in session
        session['analysis_results'] = results
        
        return jsonify({
            'success': True,
            'message': 'Analysis completed successfully',
            'analysis_id': results['analysis_id']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/results')
def get_results():
    """Get analysis results"""
    if 'analysis_results' in session:
        return jsonify(session['analysis_results'])
    else:
        # Return demo results
        demo_data()
        return jsonify(session.get('analysis_results', {}))

@app.route('/api/demo')
def run_demo():
    """Run demo analysis with sample data"""
    demo_data()
    return jsonify({
        'success': True,
        'message': 'Demo analysis completed',
        'results_available': True
    })

def demo_data():
    """Generate demo data for testing"""
    print("Generating demo analysis...")
    
    # Demo baby info
    baby_info = {
        'name': 'Alex Johnson',
        'birth_date': '2024-01-15',
        'birth_weight': 3.1,
        'apgar_1min': 7,
        'apgar_5min': 8,
        'family_history': 'Maternal cousin with autism'
    }
    
    # Run analysis with simulated data
    health_data = {
        'birth_weight': baby_info['birth_weight'],
        'apgar_1min': baby_info['apgar_1min'],
        'apgar_5min': baby_info['apgar_5min'],
        'family_history': baby_info['family_history']
    }
    
    video_analysis = ai_analyzer.analyze_video('demo_video.mp4')
    audio_analysis = ai_analyzer.analyze_audio('demo_audio.wav')
    health_analysis = ai_analyzer.analyze_health_data(health_data)
    risk_scores = ai_analyzer.calculate_risk_scores(video_analysis, audio_analysis, health_analysis)
    recommendations = ai_analyzer.generate_recommendations(risk_scores)
    
    results = {
        'baby_info': baby_info,
        'video_analysis': video_analysis,
        'audio_analysis': audio_analysis,
        'health_analysis': health_analysis,
        'risk_scores': risk_scores,
        'recommendations': recommendations,
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'analysis_id': f"NM{random.randint(10000, 99999)}"
    }
    
    session['analysis_results'] = results
    session['baby_info'] = baby_info
    return results

@app.route('/api/generate_report')
def generate_report():
    """Generate a PDF report (simulated)"""
    if 'analysis_results' not in session:
        demo_data()
    
    # Create a simple text report
    results = session['analysis_results']
    report = f"""
    NEO MIND - DEVELOPMENTAL RISK ASSESSMENT REPORT
    ================================================
    
    Baby Information:
    - Name: {results['baby_info']['name']}
    - Birth Date: {results['baby_info']['birth_date']}
    - Birth Weight: {results['baby_info']['birth_weight']} kg
    - Apgar Scores: {results['baby_info']['apgar_1min']} (1-min), {results['baby_info']['apgar_5min']} (5-min)
    
    Risk Assessment Summary:
    - Overall Risk Level: {results['risk_scores']['overall_risk']}
    
    Disorder-Specific Risk Scores:
    """
    
    for disorder, score in results['risk_scores']['risk_scores'].items():
        report += f"    - {disorder}: {score:.1%} risk\n"
    
    report += f"""
    
    Key Findings:
    - Video Analysis: {len(results['video_analysis']['risk_indicators'])} behavioral markers
    - Audio Analysis: {len(results['audio_analysis']['risk_indicators'])} vocal pattern concerns
    - Health Factors: {results['health_analysis']['total_risk_factors']} risk factors identified
    
    Recommendations:
    """
    
    for rec in results['recommendations']['general']:
        report += f"    - {rec}\n"
    
    report += f"""
    
    Report Generated: {results['analysis_date']}
    Analysis ID: {results['analysis_id']}
    
    *** This is a screening tool, not a diagnostic device ***
    *** Always consult with healthcare professionals ***
    """
    
    return jsonify({
        'report_content': report,
        'filename': f"NeoMind_Report_{results['analysis_id']}.txt"
    })

# ========== HTML Templates ==========
def create_html_templates():
    """Create HTML templates for the web interface"""
    
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeoMind - Early Detection Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-color: #4a6fa5;
            --secondary-color: #166088;
            --accent-color: #17a2b8;
            --light-bg: #f8f9fa;
            --risk-low: #28a745;
            --risk-medium: #ffc107;
            --risk-high: #dc3545;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .nav-tabs .nav-link.active {
            background: var(--primary-color);
            color: white;
            border-radius: 10px;
        }
        .risk-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .risk-low { background: var(--risk-low); color: white; }
        .risk-medium { background: var(--risk-medium); color: black; }
        .risk-high { background: var(--risk-high); color: white; }
        .feature-icon {
            font-size: 2.5rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        .disorder-card {
            transition: transform 0.3s;
            cursor: pointer;
        }
        .disorder-card:hover {
            transform: translateY(-5px);
        }
        .progress-bar-risk {
            transition: width 1s ease-in-out;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="glass-card p-4 mb-4">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1 class="display-4 text-primary">
                        <i class="fas fa-brain me-3"></i>NeoMind
                    </h1>
                    <p class="lead text-muted">AI-Powered Early Detection of Neurodevelopmental Disorders</p>
                    <p class="text-muted">Analyzing subtle behavioral, physiological, and genetic markers for early intervention</p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="badge bg-info fs-6 p-3">Prototype v1.0</div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-4 mb-4">
                <div class="glass-card h-100 p-4">
                    <div class="feature-icon text-center">
                        <i class="fas fa-video"></i>
                    </div>
                    <h4 class="text-center">Video Analysis</h4>
                    <p>Computer vision analyzes movements, facial expressions, and eye-tracking patterns</p>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item">Eye contact detection</li>
                        <li class="list-group-item">Motor coordination</li>
                        <li class="list-group-item">Facial expressivity</li>
                    </ul>
                </div>
            </div>

            <div class="col-md-4 mb-4">
                <div class="glass-card h-100 p-4">
                    <div class="feature-icon text-center">
                        <i class="fas fa-wave-square"></i>
                    </div>
                    <h4 class="text-center">Audio Analysis</h4>
                    <p>AI analyzes crying patterns, vocalizations, and babbling for atypical traits</p>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item">Cry pattern analysis</li>
                        <li class="list-group-item">Vocal complexity</li>
                        <li class="list-group-item">Pitch variability</li>
                    </ul>
                </div>
            </div>

            <div class="col-md-4 mb-4">
                <div class="glass-card h-100 p-4">
                    <div class="feature-icon text-center">
                        <i class="fas fa-dna"></i>
                    </div>
                    <h4 class="text-center">Health Data</h4>
                    <p>Comprehensive analysis of birth records, Apgar scores, and family history</p>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item">Birth metrics</li>
                        <li class="list-group-item">Family medical history</li>
                        <li class="list-group-item">Genetic risk factors</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="glass-card p-4 mb-4">
            <h3 class="mb-4">Quick Actions</h3>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <a href="/upload" class="btn btn-primary btn-lg w-100 py-3">
                        <i class="fas fa-upload me-2"></i>Upload Data & Analyze
                    </a>
                </div>
                <div class="col-md-6 mb-3">
                    <button onclick="runDemo()" class="btn btn-info btn-lg w-100 py-3">
                        <i class="fas fa-play-circle me-2"></i>Run Demo Analysis
                    </button>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <a href="/analyze" class="btn btn-success btn-lg w-100 py-3">
                        <i class="fas fa-chart-bar me-2"></i>View Analysis Results
                    </a>
                </div>
                <div class="col-md-6 mb-3">
                    <button onclick="generateReport()" class="btn btn-warning btn-lg w-100 py-3">
                        <i class="fas fa-file-pdf me-2"></i>Generate Report
                    </button>
                </div>
            </div>
        </div>

        <div class="glass-card p-4">
            <h3 class="mb-4">How It Works</h3>
            <div class="row">
                <div class="col-md-3 text-center mb-3">
                    <div class="bg-primary text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 60px; height: 60px;">
                        <i class="fas fa-database fa-lg"></i>
                    </div>
                    <h5 class="mt-2">1. Data Collection</h5>
                    <p>Upload videos, audio recordings, and health information</p>
                </div>
                <div class="col-md-3 text-center mb-3">
                    <div class="bg-primary text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 60px; height: 60px;">
                        <i class="fas fa-robot fa-lg"></i>
                    </div>
                    <h5 class="mt-2">2. AI Analysis</h5>
                    <p>Multiple AI models analyze behavioral and physiological markers</p>
                </div>
                <div class="col-md-3 text-center mb-3">
                    <div class="bg-primary text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 60px; height: 60px;">
                        <i class="fas fa-chart-line fa-lg"></i>
                    </div>
                    <h5 class="mt-2">3. Risk Assessment</h5>
                    <p>Calculate risk scores for multiple neurodevelopmental disorders</p>
                </div>
                <div class="col-md-3 text-center mb-3">
                    <div class="bg-primary text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 60px; height: 60px;">
                        <i class="fas fa-clipboard-list fa-lg"></i>
                    </div>
                    <h5 class="mt-2">4. Recommendations</h5>
                    <p>Get personalized intervention strategies and follow-up plans</p>
                </div>
            </div>
        </div>

        <div class="glass-card p-4 mt-4">
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                <strong>Note:</strong> This is a prototype for demonstration purposes. Always consult healthcare professionals for medical advice.
            </div>
        </div>
    </div>

    <script>
        function runDemo() {
            fetch('/api/demo')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Demo analysis completed! Redirecting to results...');
                        window.location.href = '/analyze';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error running demo');
                });
        }

        function generateReport() {
            fetch('/api/generate_report')
                .then(response => response.json())
                .then(data => {
                    // Create and download report file
                    const element = document.createElement('a');
                    const file = new Blob([data.report_content], {type: 'text/plain'});
                    element.href = URL.createObjectURL(file);
                    element.download = data.filename;
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                    
                    alert('Report generated and downloaded!');
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error generating report');
                });
        }
    </script>
</body>
</html>''')
    
    # Create upload.html
    with open(os.path.join(templates_dir, 'upload.html'), 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Data - NeoMind</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .upload-area {
            border: 3px dashed #4a6fa5;
            border-radius: 15px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: rgba(74, 111, 165, 0.1);
            border-color: #166088;
        }
        .upload-area i {
            font-size: 4rem;
            color: #4a6fa5;
            margin-bottom: 1rem;
        }
        .form-control:focus {
            border-color: #4a6fa5;
            box-shadow: 0 0 0 0.25rem rgba(74, 111, 165, 0.25);
        }
        .btn-primary {
            background: linear-gradient(135deg, #4a6fa5 0%, #166088 100%);
            border: none;
            padding: 12px 30px;
            font-size: 1.1rem;
        }
        .btn-primary:hover {
            background: linear-gradient(135deg, #166088 0%, #4a6fa5 100%);
        }
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2rem;
        }
        .step {
            flex: 1;
            text-align: center;
            position: relative;
        }
        .step-number {
            width: 40px;
            height: 40px;
            background: #e9ecef;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .step.active .step-number {
            background: #4a6fa5;
            color: white;
        }
        .step-line {
            position: absolute;
            top: 20px;
            left: 50%;
            right: 50%;
            height: 2px;
            background: #e9ecef;
            z-index: -1;
        }
        .step:first-child .step-line { left: 50%; right: 0; }
        .step:last-child .step-line { left: 0; right: 50%; }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="glass-card p-4 mb-4">
            <h1 class="text-primary mb-4">
                <a href="/" class="text-decoration-none text-primary">
                    <i class="fas fa-arrow-left me-3"></i>
                </a>
                Upload Baby Data for Analysis
            </h1>
            <p class="text-muted">Please provide the following information for comprehensive analysis</p>
        </div>

        <div class="step-indicator">
            <div class="step active">
                <div class="step-number">1</div>
                <div>Baby Information</div>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <div>Upload Media</div>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <div>Analysis</div>
            </div>
        </div>

        <form id="uploadForm" enctype="multipart/form-data">
            <div class="glass-card p-4 mb-4">
                <h3 class="mb-4"><i class="fas fa-baby me-2"></i>Baby Information</h3>
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Baby's Name</label>
                        <input type="text" class="form-control" name="baby_name" value="Alex Johnson" required>
                    </div>
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Date of Birth</label>
                        <input type="date" class="form-control" name="birth_date" value="2024-01-15" required>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-4 mb-3">
                        <label class="form-label">Birth Weight (kg)</label>
                        <input type="number" step="0.1" class="form-control" name="birth_weight" value="3.2" required>
                    </div>
                    <div class="col-md-4 mb-3">
                        <label class="form-label">Apgar Score (1 min)</label>
                        <input type="number" min="0" max="10" class="form-control" name="apgar_1min" value="8" required>
                    </div>
                    <div class="col-md-4 mb-3">
                        <label class="form-label">Apgar Score (5 min)</label>
                        <input type="number" min="0" max="10" class="form-control" name="apgar_5min" value="9" required>
                    </div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Family Medical History</label>
                    <textarea class="form-control" name="family_history" rows="3">Maternal cousin with autism spectrum disorder</textarea>
                    <div class="form-text">Include any family history of neurodevelopmental disorders, genetic conditions, or developmental delays.</div>
                </div>
            </div>

            <div class="glass-card p-4 mb-4">
                <h3 class="mb-4"><i class="fas fa-video me-2"></i>Video Upload</h3>
                <div class="upload-area" onclick="document.getElementById('videoFile').click()">
                    <i class="fas fa-video"></i>
                    <h4>Upload Baby Video</h4>
                    <p class="text-muted">Click to upload or drag and drop</p>
                    <p class="text-muted">Recommended: 30-60 second video showing baby's face and movements</p>
                    <input type="file" id="videoFile" name="video" accept="video/*" style="display: none;" onchange="updateFileName('videoFile', 'videoName')">
                    <div id="videoName" class="text-info mt-2">No file selected</div>
                </div>
                <div class="alert alert-info mt-3">
                    <i class="fas fa-info-circle me-2"></i>
                    Video analysis will examine: Eye contact, facial expressions, motor movements, and gaze following.
                </div>
            </div>

            <div class="glass-card p-4 mb-4">
                <h3 class="mb-4"><i class="fas fa-wave-square me-2"></i>Audio Upload</h3>
                <div class="upload-area" onclick="document.getElementById('audioFile').click()">
                    <i class="fas fa-wave-square"></i>
                    <h4>Upload Baby Audio</h4>
                    <p class="text-muted">Click to upload or drag and drop</p>
                    <p class="text-muted">Recommended: Recording of baby's cries, coos, or vocalizations</p>
                    <input type="file" id="audioFile" name="audio" accept="audio/*" style="display: none;" onchange="updateFileName('audioFile', 'audioName')">
                    <div id="audioName" class="text-info mt-2">No file selected</div>
                </div>
                <div class="alert alert-info mt-3">
                    <i class="fas fa-info-circle me-2"></i>
                    Audio analysis will examine: Cry patterns, pitch variability, vocal complexity, and rhythm consistency.
                </div>
            </div>

            <div class="glass-card p-4 mb-4">
                <h3 class="mb-4"><i class="fas fa-shield-alt me-2"></i>Privacy & Consent</h3>
                <div class="form-check mb-3">
                    <input class="form-check-input" type="checkbox" id="consentCheck" required>
                    <label class="form-check-label" for="consentCheck">
                        I consent to the analysis of this data for developmental screening purposes.
                    </label>
                </div>
                <div class="form-check mb-3">
                    <input class="form-check-input" type="checkbox" id="privacyCheck" required>
                    <label class="form-check-label" for="privacyCheck">
                        I acknowledge that this data will be processed securely and anonymously.
                    </label>
                </div>
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Important:</strong> This tool is for screening purposes only. It is not a diagnostic device. Always consult with healthcare professionals for medical advice.
                </div>
            </div>

            <div class="text-center">
                <button type="button" onclick="submitForm()" class="btn btn-primary btn-lg px-5">
                    <i class="fas fa-play-circle me-2"></i>Start Analysis
                </button>
                <button type="button" onclick="window.location.href='/'" class="btn btn-outline-secondary btn-lg px-5 ms-3">
                    Cancel
                </button>
            </div>
        </form>

        <div class="glass-card p-4 mt-4">
            <h4><i class="fas fa-lightbulb me-2"></i>Tips for Best Results</h4>
            <ul class="mb-0">
                <li>Record video in good lighting with baby's face clearly visible</li>
                <li>Capture a variety of movements and interactions</li>
                <li>Record audio in a quiet environment</li>
                <li>Provide complete and accurate health information</li>
                <li>Include any relevant family medical history</li>
            </ul>
        </div>
    </div>

    <div class="modal fade" id="analysisModal" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Analysis in Progress</h5>
                </div>
                <div class="modal-body text-center">
                    <div class="spinner-border text-primary mb-3" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h4>Analyzing Baby Data</h4>
                    <p>This may take a few moments...</p>
                    <div class="progress mt-3">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 100%"></div>
                    </div>
                    <div class="mt-3">
                        <div class="d-flex justify-content-between">
                            <small>Video Analysis</small>
                            <small><i class="fas fa-check text-success"></i></small>
                        </div>
                        <div class="d-flex justify-content-between">
                            <small>Audio Analysis</small>
                            <small><i class="fas fa-check text-success"></i></small>
                        </div>
                        <div class="d-flex justify-content-between">
                            <small>Health Data Analysis</small>
                            <small><i class="fas fa-check text-success"></i></small>
                        </div>
                        <div class="d-flex justify-content-between">
                            <small>Risk Assessment</small>
                            <small><i class="fas fa-check text-success"></i></small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateFileName(inputId, displayId) {
            const input = document.getElementById(inputId);
            const display = document.getElementById(displayId);
            if (input.files.length > 0) {
                display.textContent = input.files[0].name;
                display.className = 'text-success mt-2';
            }
        }

        function submitForm() {
            if (!document.getElementById('consentCheck').checked || 
                !document.getElementById('privacyCheck').checked) {
                alert('Please agree to the privacy and consent statements');
                return;
            }

            const form = document.getElementById('uploadForm');
            const formData = new FormData(form);
            
            const modal = new bootstrap.Modal(document.getElementById('analysisModal'));
            modal.show();

            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    setTimeout(() => {
                        modal.hide();
                        alert('Analysis completed successfully!');
                        window.location.href = '/analyze';
                    }, 2000);
                } else {
                    modal.hide();
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                modal.hide();
                alert('Error submitting form');
                console.error('Error:', error);
            });
        }

        // Allow drag and drop for upload areas
        const uploadAreas = document.querySelectorAll('.upload-area');
        uploadAreas.forEach(area => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.style.backgroundColor = 'rgba(74, 111, 165, 0.1)';
            });

            area.addEventListener('dragleave', (e) => {
                e.preventDefault();
                area.style.backgroundColor = '';
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.style.backgroundColor = '';
                const files = e.dataTransfer.files;
                // Handle dropped files here
                console.log('Files dropped:', files);
            });
        });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>''')
    
    # Create analyze.html
    with open(os.path.join(templates_dir, 'analyze.html'), 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - NeoMind</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .risk-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .risk-low { background: #28a745; color: white; }
        .risk-medium { background: #ffc107; color: black; }
        .risk-high { background: #dc3545; color: white; }
        .progress-bar-risk {
            transition: width 1s ease-in-out;
            height: 25px;
            border-radius: 12px;
        }
        .disorder-card {
            transition: transform 0.3s;
            border-left: 4px solid;
        }
        .disorder-card:hover {
            transform: translateY(-3px);
        }
        .metric-card {
            text-align: center;
            padding: 1.5rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4a6fa5;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        .tab-pane {
            padding: 1.5rem 0;
        }
        .nav-tabs .nav-link {
            border: none;
            color: #666;
            font-weight: 500;
        }
        .nav-tabs .nav-link.active {
            color: #4a6fa5;
            border-bottom: 3px solid #4a6fa5;
            background: transparent;
        }
        .recommendation-item {
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 10px;
            border-left: 4px solid #4a6fa5;
            background: #f8f9fa;
        }
        .priority-high {
            border-left-color: #dc3545;
        }
        .priority-medium {
            border-left-color: #ffc107;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="glass-card p-4 mb-4">
            <h1 class="text-primary mb-4">
                <a href="/" class="text-decoration-none text-primary">
                    <i class="fas fa-arrow-left me-3"></i>
                </a>
                Analysis Results
                <span id="analysisStatus" class="badge bg-info float-end mt-2">Processing Complete</span>
            </h1>
            <div id="babyInfo" class="row">
                <!-- Baby info will be loaded here -->
            </div>
        </div>

        <ul class="nav nav-tabs mb-4" id="resultsTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button">
                    <i class="fas fa-chart-pie me-2"></i>Overview
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="risks-tab" data-bs-toggle="tab" data-bs-target="#risks" type="button">
                    <i class="fas fa-exclamation-triangle me-2"></i>Risk Assessment
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="details-tab" data-bs-toggle="tab" data-bs-target="#details" type="button">
                    <i class="fas fa-chart-line me-2"></i>Detailed Analysis
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="recommendations-tab" data-bs-toggle="tab" data-bs-target="#recommendations" type="button">
                    <i class="fas fa-clipboard-list me-2"></i>Recommendations
                </button>
            </li>
        </ul>

        <div class="tab-content" id="resultsTabContent">
            <!-- Overview Tab -->
            <div class="tab-pane fade show active" id="overview" role="tabpanel">
                <div class="row">
                    <div class="col-md-8">
                        <div class="glass-card p-4 h-100">
                            <h4 class="mb-4">Risk Distribution</h4>
                            <canvas id="riskChart" height="200"></canvas>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="glass-card p-4 h-100">
                            <h4 class="mb-4">Overall Risk Level</h4>
                            <div class="text-center">
                                <div id="overallRiskBadge" class="risk-badge risk-medium d-inline-block mb-3" style="font-size: 1.2rem;">Medium Risk</div>
                                <div class="mb-4">
                                    <div class="metric-value" id="overallRiskScore">0.45</div>
                                    <div class="metric-label">Composite Risk Score</div>
                                </div>
                                <div class="alert alert-info">
                                    <i class="fas fa-info-circle me-2"></i>
                                    Based on analysis of behavioral, vocal, and health markers
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row mt-4">
                    <div class="col-md-3">
                        <div class="glass-card metric-card">
                            <div class="metric-value" id="videoMarkers">0</div>
                            <div class="metric-label">Behavioral Markers</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="glass-card metric-card">
                            <div class="metric-value" id="audioMarkers">0</div>
                            <div class="metric-label">Vocal Concerns</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="glass-card metric-card">
                            <div class="metric-value" id="healthFactors">0</div>
                            <div class="metric-label">Risk Factors</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="glass-card metric-card">
                            <div class="metric-value" id="recommendationCount">0</div>
                            <div class="metric-label">Recommendations</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Risk Assessment Tab -->
            <div class="tab-pane fade" id="risks" role="tabpanel">
                <div class="glass-card p-4 mb-4">
                    <h4 class="mb-4">Disorder-Specific Risk Assessment</h4>
                    <div id="riskCards">
                        <!-- Risk cards will be loaded here -->
                    </div>
                </div>

                <div class="row">
                    <div class="col-md-6">
                        <div class="glass-card p-4 h-100">
                            <h5 class="mb-3">Risk Components</h5>
                            <div id="componentDetails">
                                <!-- Component details will be loaded here -->
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card p-4 h-100">
                            <h5 class="mb-3">Timeline</h5>
                            <div class="timeline">
                                <div class="d-flex mb-3">
                                    <div class="flex-shrink-0">
                                        <div class="bg-primary text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 40px; height: 40px;">
                                            <i class="fas fa-calendar-check"></i>
                                        </div>
                                    </div>
                                    <div class="flex-grow-1 ms-3">
                                        <h6>Immediate (1-2 weeks)</h6>
                                        <p class="mb-0">Consult with pediatrician about findings</p>
                                    </div>
                                </div>
                                <div class="d-flex mb-3">
                                    <div class="flex-shrink-0">
                                        <div class="bg-info text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 40px; height: 40px;">
                                            <i class="fas fa-stethoscope"></i>
                                        </div>
                                    </div>
                                    <div class="flex-grow-1 ms-3">
                                        <h6>Short-term (1 month)</h6>
                                        <p class="mb-0">Developmental milestone assessment</p>
                                    </div>
                                </div>
                                <div class="d-flex">
                                    <div class="flex-shrink-0">
                                        <div class="bg-success text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 40px; height: 40px;">
                                            <i class="fas fa-heartbeat"></i>
                                        </div>
                                    </div>
                                    <div class="flex-grow-1 ms-3">
                                        <h6>Follow-up (3-6 months)</h6>
                                        <p class="mb-0">Re-evaluation and progress tracking</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detailed Analysis Tab -->
            <div class="tab-pane fade" id="details" role="tabpanel">
                <div class="glass-card p-4 mb-4">
                    <h4 class="mb-4">Video Analysis Details</h4>
                    <div id="videoDetails">
                        <!-- Video analysis details will be loaded here -->
                    </div>
                </div>

                <div class="glass-card p-4 mb-4">
                    <h4 class="mb-4">Audio Analysis Details</h4>
                    <div id="audioDetails">
                        <!-- Audio analysis details will be loaded here -->
                    </div>
                </div>

                <div class="glass-card p-4">
                    <h4 class="mb-4">Health Data Analysis</h4>
                    <div id="healthDetails">
                        <!-- Health analysis details will be loaded here -->
                    </div>
                </div>
            </div>

            <!-- Recommendations Tab -->
            <div class="tab-pane fade" id="recommendations" role="tabpanel">
                <div class="glass-card p-4 mb-4">
                    <h4 class="mb-4">Personalized Recommendations</h4>
                    <div id="specificRecommendations">
                        <!-- Disorder-specific recommendations will be loaded here -->
                    </div>
                </div>

                <div class="glass-card p-4 mb-4">
                    <h4 class="mb-4">General Recommendations</h4>
                    <div id="generalRecommendations">
                        <!-- General recommendations will be loaded here -->
                    </div>
                </div>

                <div class="glass-card p-4">
                    <div class="row">
                        <div class="col-md-6">
                            <h5 class="mb-3"><i class="fas fa-download me-2"></i>Export Report</h5>
                            <p>Download a comprehensive report of the analysis for sharing with healthcare providers.</p>
                            <button onclick="generateReport()" class="btn btn-primary">
                                <i class="fas fa-file-pdf me-2"></i>Generate PDF Report
                            </button>
                        </div>
                        <div class="col-md-6">
                            <h5 class="mb-3"><i class="fas fa-share-alt me-2"></i>Share Results</h5>
                            <p>Share these results with your pediatrician or specialist for further evaluation.</p>
                            <button onclick="shareResults()" class="btn btn-outline-primary">
                                <i class="fas fa-share me-2"></i>Share with Doctor
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="glass-card p-4 mt-4">
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Important Disclaimer:</strong> This tool is for screening purposes only. It is not a diagnostic device. The results should be discussed with qualified healthcare professionals. Always seek professional medical advice for diagnosis and treatment.
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let resultsData = null;
        let riskChart = null;

        document.addEventListener('DOMContentLoaded', function() {
            loadResults();
        });

        function loadResults() {
            fetch('/api/results')
                .then(response => response.json())
                .then(data => {
                    resultsData = data;
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error loading results:', error);
                    // Fallback to demo data
                    resultsData = getDemoData();
                    displayResults(resultsData);
                });
        }

        function displayResults(data) {
            // Update baby info
            const babyInfo = data.baby_info;
            document.getElementById('babyInfo').innerHTML = `
                <div class="col-md-6">
                    <h5>${babyInfo.name}</h5>
                    <p class="text-muted mb-1">Date of Birth: ${babyInfo.birth_date}</p>
                    <p class="text-muted mb-1">Birth Weight: ${babyInfo.birth_weight} kg</p>
                </div>
                <div class="col-md-6">
                    <p class="text-muted mb-1">Apgar Scores: ${babyInfo.apgar_1min} (1-min), ${babyInfo.apgar_5min} (5-min)</p>
                    <p class="text-muted mb-1">Family History: ${babyInfo.family_history}</p>
                    <p class="text-muted mb-0">Analysis Date: ${data.analysis_date}</p>
                </div>
            `;

            // Update metrics
            document.getElementById('videoMarkers').textContent = data.video_analysis.risk_indicators.length;
            document.getElementById('audioMarkers').textContent = data.audio_analysis.risk_indicators.length;
            document.getElementById('healthFactors').textContent = data.health_analysis.total_risk_factors;
            document.getElementById('recommendationCount').textContent = 
                data.recommendations.disorder_specific.length + data.recommendations.general.length;

            // Update overall risk
            const overallRisk = data.risk_scores.overall_risk;
            const overallRiskBadge = document.getElementById('overallRiskBadge');
            overallRiskBadge.textContent = `${overallRisk} Risk`;
            overallRiskBadge.className = `risk-badge risk-${overallRisk.toLowerCase()} d-inline-block mb-3`;
            
            // Calculate average risk score for display
            const riskScores = Object.values(data.risk_scores.risk_scores);
            const avgRisk = (riskScores.reduce((a, b) => a + b, 0) / riskScores.length).toFixed(3);
            document.getElementById('overallRiskScore').textContent = avgRisk;

            // Create risk chart
            createRiskChart(data.risk_scores.risk_scores);

            // Display risk cards
            displayRiskCards(data.risk_scores);

            // Display component details
            displayComponentDetails(data.risk_scores);

            // Display video details
            displayVideoDetails(data.video_analysis);

            // Display audio details
            displayAudioDetails(data.audio_analysis);

            // Display health details
            displayHealthDetails(data.health_analysis);

            // Display recommendations
            displayRecommendations(data.recommendations);
        }

        function createRiskChart(riskScores) {
            const ctx = document.getElementById('riskChart').getContext('2d');
            
            if (riskChart) {
                riskChart.destroy();
            }

            const disorders = Object.keys(riskScores);
            const scores = Object.values(riskScores);
            const backgroundColors = scores.map(score => {
                if (score > 0.6) return '#dc3545';
                if (score > 0.3) return '#ffc107';
                return '#28a745';
            });

            riskChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: disorders,
                    datasets: [{
                        label: 'Risk Score',
                        data: scores,
                        backgroundColor: backgroundColors,
                        borderColor: backgroundColors.map(color => color.replace('0.8', '1')),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                callback: function(value) {
                                    return (value * 100) + '%';
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.label}: ${(context.raw * 100).toFixed(1)}% risk`;
                                }
                            }
                        }
                    }
                }
            });
        }

        function displayRiskCards(riskData) {
            const riskCardsDiv = document.getElementById('riskCards');
            let html = '<div class="row">';
            
            Object.entries(riskData.risk_scores).forEach(([disorder, score]) => {
                let riskLevel = 'Low';
                let riskClass = 'risk-low';
                if (score > 0.6) {
                    riskLevel = 'High';
                    riskClass = 'risk-high';
                } else if (score > 0.3) {
                    riskLevel = 'Medium';
                    riskClass = 'risk-medium';
                }

                const percent = Math.round(score * 100);
                
                html += `
                <div class="col-md-6 mb-3">
                    <div class="disorder-card glass-card p-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h5 class="mb-0">${disorder}</h5>
                            <span class="risk-badge ${riskClass}">${riskLevel} Risk</span>
                        </div>
                        <p class="text-muted small mb-3">${riskData.explanations[disorder]}</p>
                        <div class="progress mb-2">
                            <div class="progress-bar progress-bar-risk ${riskClass.replace('risk-', 'bg-')}" 
                                 role="progressbar" style="width: ${percent}%">
                                ${percent}%
                            </div>
                        </div>
                        <div class="text-end">
                            <small class="text-muted">Risk Score: ${score.toFixed(3)}</small>
                        </div>
                    </div>
                </div>`;
            });
            
            html += '</div>';
            riskCardsDiv.innerHTML = html;
        }

        function displayComponentDetails(riskData) {
            const componentDiv = document.getElementById('componentDetails');
            const components = riskData.components;
            
            let html = `
                <div class="mb-3">
                    <label class="form-label">Video Analysis Risk: ${components.video_risk}</label>
                    <div class="progress mb-2">
                        <div class="progress-bar bg-info" style="width: ${components.video_risk * 100}%"></div>
                    </div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Audio Analysis Risk: ${components.audio_risk}</label>
                    <div class="progress mb-2">
                        <div class="progress-bar bg-warning" style="width: ${components.audio_risk * 100}%"></div>
                    </div>
                </div>
                <div class="mb-3">
                    <label class="form-label">Health Data Risk: ${components.health_risk}</label>
                    <div class="progress mb-2">
                        <div class="progress-bar bg-danger" style="width: ${components.health_risk * 100}%"></div>
                    </div>
                </div>
            `;
            
            componentDiv.innerHTML = html;
        }

        function displayVideoDetails(videoAnalysis) {
            const videoDiv = document.getElementById('videoDetails');
            let html = '<div class="row">';
            
            // Display metrics
            Object.entries(videoAnalysis.metrics).forEach(([metric, value]) => {
                const percent = Math.round(value * 100);
                let barColor = 'bg-success';
                if (value < 0.4) barColor = 'bg-danger';
                else if (value < 0.6) barColor = 'bg-warning';
                
                html += `
                <div class="col-md-6 mb-3">
                    <div class="d-flex justify-content-between">
                        <span>${metric.replace('_', ' ').toUpperCase()}</span>
                        <span>${value.toFixed(2)}</span>
                    </div>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar ${barColor}" style="width: ${percent}%"></div>
                    </div>
                </div>`;
            });
            
            html += '</div>';
            
            // Display risk indicators
            if (videoAnalysis.risk_indicators.length > 0) {
                html += `
                <div class="alert alert-warning mt-3">
                    <h6><i class="fas fa-exclamation-triangle me-2"></i>Behavioral Markers Detected:</h6>
                    <ul class="mb-0">`;
                
                videoAnalysis.risk_indicators.forEach(indicator => {
                    html += `<li>${indicator}</li>`;
                });
                
                html += `</ul></div>`;
            } else {
                html += `
                <div class="alert alert-success mt-3">
                    <i class="fas fa-check-circle me-2"></i>
                    No significant behavioral markers detected in video analysis.
                </div>`;
            }
            
            videoDiv.innerHTML = html;
        }

        function displayAudioDetails(audioAnalysis) {
            const audioDiv = document.getElementById('audioDetails');
            let html = '<div class="row">';
            
            // Display metrics
            Object.entries(audioAnalysis.metrics).forEach(([metric, value]) => {
                if (typeof value === 'number') {
                    const percent = Math.min(Math.round(value * 100), 100);
                    let barColor = 'bg-success';
                    if (value < 0.3) barColor = 'bg-danger';
                    else if (value < 0.5) barColor = 'bg-warning';
                    
                    html += `
                    <div class="col-md-6 mb-3">
                        <div class="d-flex justify-content-between">
                            <span>${metric.replace('_', ' ').toUpperCase()}</span>
                            <span>${value.toFixed(2)}</span>
                        </div>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar ${barColor}" style="width: ${percent}%"></div>
                        </div>
                    </div>`;
                }
            });
            
            html += '</div>';
            
            // Display risk indicators
            if (audioAnalysis.risk_indicators.length > 0) {
                html += `
                <div class="alert alert-warning mt-3">
                    <h6><i class="fas fa-exclamation-triangle me-2"></i>Vocal Pattern Concerns:</h6>
                    <ul class="mb-0">`;
                
                audioAnalysis.risk_indicators.forEach(indicator => {
                    html += `<li>${indicator}</li>`;
                });
                
                html += `</ul></div>`;
            } else {
                html += `
                <div class="alert alert-success mt-3">
                    <i class="fas fa-check-circle me-2"></i>
                    No significant vocal pattern concerns detected.
                </div>`;
            }
            
            audioDiv.innerHTML = html;
        }

        function displayHealthDetails(healthAnalysis) {
            const healthDiv = document.getElementById('healthDetails');
            let html = '<div class="row">';
            
            // Display risk factors
            html += `
            <div class="col-md-6">
                <div class="card border-danger mb-3">
                    <div class="card-header bg-danger text-white">
                        <i class="fas fa-exclamation-circle me-2"></i>Risk Factors (${healthAnalysis.risk_factors.length})
                    </div>
                    <div class="card-body">`;
            
            if (healthAnalysis.risk_factors.length > 0) {
                html += '<ul class="mb-0">';
                healthAnalysis.risk_factors.forEach(factor => {
                    html += `<li>${factor}</li>`;
                });
                html += '</ul>';
            } else {
                html += '<p class="mb-0 text-muted">No significant risk factors identified.</p>';
            }
            
            html += `</div></div></div>`;
            
            // Display protective factors
            html += `
            <div class="col-md-6">
                <div class="card border-success mb-3">
                    <div class="card-header bg-success text-white">
                        <i class="fas fa-shield-alt me-2"></i>Protective Factors (${healthAnalysis.protective_factors.length})
                    </div>
                    <div class="card-body">`;
            
            if (healthAnalysis.protective_factors.length > 0) {
                html += '<ul class="mb-0">';
                healthAnalysis.protective_factors.forEach(factor => {
                    html += `<li>${factor}</li>`;
                });
                html += '</ul>';
            } else {
                html += '<p class="mb-0 text-muted">No protective factors identified.</p>';
            }
            
            html += `</div></div></div>`;
            
            html += '</div>';
            healthDiv.innerHTML = html;
        }

        function displayRecommendations(recommendations) {
            // Display disorder-specific recommendations
            const specificDiv = document.getElementById('specificRecommendations');
            let specificHtml = '';
            
            if (recommendations.disorder_specific.length > 0) {
                recommendations.disorder_specific.forEach(rec => {
                    specificHtml += `
                    <div class="recommendation-item priority-${rec.priority.toLowerCase()} mb-3">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h6 class="mb-1">${rec.disorder} - ${rec.priority} Priority</h6>
                                <ul class="mb-0 ps-3">`;
                    
                    rec.actions.forEach(action => {
                        specificHtml += `<li>${action}</li>`;
                    });
                    
                    specificHtml += `
                                </ul>
                            </div>
                            <span class="badge bg-${rec.priority === 'High' ? 'danger' : 'warning'}">
                                ${rec.priority}
                            </span>
                        </div>
                    </div>`;
                });
            } else {
                specificHtml = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No disorder-specific high-priority recommendations at this time.
                </div>`;
            }
            
            specificDiv.innerHTML = specificHtml;
            
            // Display general recommendations
            const generalDiv = document.getElementById('generalRecommendations');
            let generalHtml = '<div class="row">';
            
            recommendations.general.forEach((rec, index) => {
                generalHtml += `
                <div class="col-md-6 mb-3">
                    <div class="d-flex">
                        <div class="flex-shrink-0">
                            <div class="bg-primary text-white rounded-circle d-inline-flex align-items-center justify-content-center" style="width: 30px; height: 30px;">
                                ${index + 1}
                            </div>
                        </div>
                        <div class="flex-grow-1 ms-3">
                            <p class="mb-0">${rec}</p>
                        </div>
                    </div>
                </div>`;
            });
            
            generalHtml += '</div>';
            generalDiv.innerHTML = generalHtml;
        }

        function generateReport() {
            fetch('/api/generate_report')
                .then(response => response.json())
                .then(data => {
                    // Create and download report file
                    const element = document.createElement('a');
                    const file = new Blob([data.report_content], {type: 'text/plain'});
                    element.href = URL.createObjectURL(file);
                    element.download = data.filename;
                    document.body.appendChild(element);
                    element.click();
                    document.body.removeChild(element);
                    
                    alert('Report generated and downloaded successfully!');
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error generating report');
                });
        }

        function shareResults() {
            if (!resultsData) return;
            
            const reportContent = `
NeoMind Analysis Results for ${resultsData.baby_info.name}
========================================================

Risk Assessment Summary:
${Object.entries(resultsData.risk_scores.risk_scores)
  .map(([disorder, score]) => `- ${disorder}: ${(score * 100).toFixed(1)}%`)
  .join('\n')}

Overall Risk Level: ${resultsData.risk_scores.overall_risk}

Key Recommendations:
${resultsData.recommendations.general.slice(0, 3).join('\n')}

Analysis ID: ${resultsData.analysis_id}
Date: ${resultsData.analysis_date}
            `.trim();

            if (navigator.share) {
                navigator.share({
                    title: 'NeoMind Analysis Results',
                    text: reportContent,
                    url: window.location.href
                });
            } else {
                // Fallback: copy to clipboard
                navigator.clipboard.writeText(reportContent)
                    .then(() => alert('Results copied to clipboard!'))
                    .catch(() => alert('Please manually copy the results.'));
            }
        }

        // Fallback demo data
        function getDemoData() {
            return {
                baby_info: {
                    name: "Alex Johnson",
                    birth_date: "2024-01-15",
                    birth_weight: 3.2,
                    apgar_1min: 8,
                    apgar_5min: 9,
                    family_history: "Maternal cousin with autism"
                },
                analysis_date: new Date().toISOString(),
                analysis_id: "NM12345",
                video_analysis: {
                    risk_indicators: ["Reduced eye contact", "Limited facial expressions"],
                    metrics: {
                        eye_contact_score: 0.35,
                        gaze_following: 0.65,
                        facial_expressivity: 0.42,
                        motor_coordination: 0.72,
                        limb_symmetry: 0.88,
                        movement_smoothness: 0.79
                    }
                },
                audio_analysis: {
                    risk_indicators: ["Monotonous vocal patterns"],
                    metrics: {
                        duration_seconds: 8.5,
                        pitch_variability: 0.22,
                        cry_rhythm_consistency: 0.68,
                        vocalization_complexity: 0.41
                    }
                },
                health_analysis: {
                    risk_factors: ["Low 1-min Apgar: 7", "Family history of autism"],
                    protective_factors: ["Normal birth weight"],
                    total_risk_factors: 2
                },
                risk_scores: {
                    risk_scores: {
                        "ASD": 0.65,
                        "ADHD": 0.42,
                        "Down Syndrome": 0.28,
                        "Developmental Delay": 0.51
                    },
                    overall_risk: "Medium",
                    explanations: {
                        "ASD": "Based on 2 behavioral markers",
                        "ADHD": "Primarily from activity and attention patterns",
                        "Down Syndrome": "Health markers and physical features analysis",
                        "Developmental Delay": "Overall developmental progress assessment"
                    },
                    components: {
                        video_risk: 0.45,
                        audio_risk: 0.32,
                        health_risk: 0.28
                    }
                },
                recommendations: {
                    disorder_specific: [
                        {
                            disorder: "ASD",
                            priority: "High",
                            actions: [
                                "Schedule consultation with pediatric neurologist",
                                "Begin early intervention assessment",
                                "Monitor specific developmental milestones weekly"
                            ]
                        },
                        {
                            disorder: "Developmental Delay",
                            priority: "Medium",
                            actions: [
                                "Discuss findings with pediatrician",
                                "Track development with milestone checklist"
                            ]
                        }
                    ],
                    general: [
                        "Maintain regular well-baby checkups",
                        "Document developmental milestones",
                        "Engage in face-to-face interaction daily",
                        "Monitor response to name and social smiling",
                        "Consider early intervention program referral"
                    ]
                }
            };
        }
    </script>
</body>
</html>''')

# ========== Create Demo Files ==========
def create_demo_files():
    """Create demo files for testing"""
    print("Creating demo files...")
    
    # Create a dummy video file (empty MP4)
    with open('demo_video.mp4', 'wb') as f:
        f.write(b'Dummy video file for demo purposes')
    
    # Create a dummy audio file (empty WAV)
    with open('demo_audio.wav', 'wb') as f:
        f.write(b'Dummy audio file for demo purposes')
    
    print("Demo files created!")

# ========== Install Required Packages ==========
def install_packages():
    """Check and install required packages"""
    import subprocess
    import sys
    
    required_packages = [
        'flask',
        'numpy',
        'pandas',
        'opencv-python',
        'librosa',
        'soundfile',
        'Pillow',
        'matplotlib',
        'scikit-learn',
        'joblib'
    ]
    
    print("Checking and installing required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').split('==')[0])
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")

# ========== Main Execution ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("NEO MIND - AI-Based Early Detection Prototype")
    print("="*60)
    
    # Install required packages
    try:
        install_packages()
    except:
        print("\nNote: Some packages may need manual installation.")
        print("Run: pip install flask numpy pandas opencv-python librosa soundfile Pillow matplotlib scikit-learn joblib")
    
    # Create HTML templates
    print("\nCreating web interface templates...")
    create_html_templates()
    
    # Create demo files
    create_demo_files()
    
    # Start the web server
    print("\n" + "="*60)
    print("Starting NeoMind Web Server...")
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
