"""
Report generator module using pre-trained language models
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np

# Import project config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ReportGenerator:
    """Report generator using pre-trained language models"""
    
    def __init__(self, model_name=None):
        """
        Initialize the report generator
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name if model_name else config.MEDICAL_LLM_MODEL
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Initialize model and tokenizer
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the model and tokenizer
        """
        try:
            print(f"Loading model {self.model_name}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Create generator pipeline
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            print(f"Model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback text generation for demonstration")
    
    def generate_report_section(self, section_name, findings=None, patient_info=None):
        """
        Generate a section of the radiology report
        
        Args:
            section_name: Name of the report section
            findings: Dictionary of findings from image analysis
            patient_info: Dictionary of patient information
            
        Returns:
            str: Generated text for the section
        """
        # Create prompt based on section
        if section_name == "Clinical Information":
            prompt = self._create_clinical_info_prompt(patient_info)
        elif section_name == "Technique":
            prompt = self._create_technique_prompt(patient_info)
        elif section_name == "Findings":
            prompt = self._create_findings_prompt(findings)
        elif section_name == "Impression":
            prompt = self._create_impression_prompt(findings)
        elif section_name == "Recommendations":
            prompt = self._create_recommendations_prompt(findings)
        else:
            prompt = f"Write a {section_name} section for a radiology report."
        
        # Generate text using model or fallback
        if self.generator is not None:
            try:
                outputs = self.generator(
                    prompt,
                    max_length=250,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                )
                
                generated_text = outputs[0]["generated_text"]
                return generated_text
            except Exception as e:
                print(f"Error generating text: {e}")
                return self._generate_fallback_text(section_name, findings, patient_info)
        else:
            return self._generate_fallback_text(section_name, findings, patient_info)
    
    def _create_clinical_info_prompt(self, patient_info):
        """
        Create prompt for Clinical Information section
        
        Args:
            patient_info: Dictionary of patient information
            
        Returns:
            str: Prompt for the model
        """
        if patient_info:
            age = patient_info.get("age", "middle-aged")
            gender = patient_info.get("gender", "adult")
            history = patient_info.get("history", "shortness of breath")
            return f"Write a Clinical Information section for a radiology report for a {age} {gender} patient with {history}."
        else:
            return "Write a Clinical Information section for a radiology report for a patient with respiratory symptoms."
    
    def _create_technique_prompt(self, patient_info):
        """
        Create prompt for Technique section
        
        Args:
            patient_info: Dictionary of patient information
            
        Returns:
            str: Prompt for the model
        """
        if patient_info and "modality" in patient_info:
            modality = patient_info["modality"]
            return f"Write a Technique section for a {modality} radiology report."
        else:
            return "Write a Technique section for a chest X-ray radiology report."
    
    def _create_findings_prompt(self, findings):
        """
        Create prompt for Findings section based on detected anomalies
        
        Args:
            findings: Dictionary of findings from image analysis
            
        Returns:
            str: Prompt for the model
        """
        if findings and "anomalies" in findings and findings["anomalies"]:
            anomaly_text = ", ".join([f"{a['label']} (confidence: {a['confidence']:.2f})" for a in findings["anomalies"]])
            return f"Write a detailed Findings section for a radiology report describing these conditions: {anomaly_text}."
        else:
            return "Write a Findings section for a normal chest X-ray radiology report."
    
    def _create_impression_prompt(self, findings):
        """
        Create prompt for Impression section
        
        Args:
            findings: Dictionary of findings from image analysis
            
        Returns:
            str: Prompt for the model
        """
        if findings and "anomalies" in findings and findings["anomalies"]:
            anomaly_text = ", ".join([a["label"] for a in findings["anomalies"]])
            return f"Write a concise Impression section for a radiology report summarizing these findings: {anomaly_text}."
        else:
            return "Write an Impression section for a normal chest X-ray radiology report."
    
    def _create_recommendations_prompt(self, findings):
        """
        Create prompt for Recommendations section
        
        Args:
            findings: Dictionary of findings from image analysis
            
        Returns:
            str: Prompt for the model
        """
        if findings and "anomalies" in findings and findings["anomalies"]:
            anomaly_text = ", ".join([a["label"] for a in findings["anomalies"]])
            return f"Write clinical Recommendations section for a radiology report based on these findings: {anomaly_text}."
        else:
            return "Write a Recommendations section for a normal chest X-ray radiology report."
    
    def _generate_fallback_text(self, section_name, findings=None, patient_info=None):
        """
        Generate fallback text for demonstration when model is not available
        
        Args:
            section_name: Name of the report section
            findings: Dictionary of findings from image analysis
            patient_info: Dictionary of patient information
            
        Returns:
            str: Generated text for the section
        """
        if section_name == "Clinical Information":
            return "Patient presented with shortness of breath and chest pain. History of hypertension and smoking."
        
        elif section_name == "Technique":
            return "PA and lateral chest radiographs were obtained using standard technique."
        
        elif section_name == "Findings":
            if findings and "anomalies" in findings and findings["anomalies"]:
                text = "The lungs are adequately inflated. "
                
                for anomaly in findings["anomalies"]:
                    if anomaly["label"] == "Cardiomegaly":
                        text += "The cardiac silhouette is enlarged, with a cardiothoracic ratio greater than 0.5, consistent with cardiomegaly. "
                    elif anomaly["label"] == "Pleural Effusion":
                        text += "There is blunting of the costophrenic angle, consistent with a small pleural effusion. "
                    elif anomaly["label"] == "Pneumonia":
                        text += "There is a focal area of consolidation in the lung field, consistent with pneumonia. "
                    elif anomaly["label"] == "Pneumothorax":
                        text += "There is a visible pleural line with absence of lung markings peripherally, consistent with pneumothorax. "
                    elif anomaly["label"] == "Lung Opacity":
                        text += "There is an area of increased opacity in the lung field. "
                    else:
                        text += f"There are findings consistent with {anomaly['label']}. "
                
                text += "The mediastinal contours are unremarkable. No evidence of acute bone abnormality."
                return text
            else:
                return "The lungs are clear without focal consolidation, pneumothorax, or pleural effusion. Heart size is normal. The mediastinal contours are unremarkable. No acute bone abnormality."
        
        elif section_name == "Impression":
            if findings and "anomalies" in findings and findings["anomalies"]:
                text = ""
                for i, anomaly in enumerate(findings["anomalies"]):
                    text += f"{i+1}. {anomaly['label']}\n"
                return text
            else:
                return "No acute cardiopulmonary abnormality."
        
        elif section_name == "Recommendations":
            if findings and "anomalies" in findings and findings["anomalies"]:
                has_pneumonia = any(a["label"] == "Pneumonia" for a in findings["anomalies"])
                has_effusion = any(a["label"] == "Pleural Effusion" for a in findings["anomalies"])
                has_pneumothorax = any(a["label"] == "Pneumothorax" for a in findings["anomalies"])
                
                text = ""
                if has_pneumonia:
                    text += "1. Clinical correlation and appropriate antibiotic therapy for pneumonia.\n"
                if has_effusion:
                    text += "2. Consider thoracentesis if clinically indicated for pleural effusion.\n"
                if has_pneumothorax:
                    text += "3. Urgent thoracic surgery consultation for management of pneumothorax.\n"
                
                if not text:
                    text = "1. Clinical correlation recommended.\n2. Consider follow-up imaging in 3-6 months to ensure stability."
                
                return text
            else:
                return "No specific follow-up imaging is recommended at this time."
        
        else:
            return f"This is a placeholder for the {section_name} section."
    
    def generate_full_report(self, findings=None, patient_info=None):
        """
        Generate a complete radiology report
        
        Args:
            findings: Dictionary of findings from image analysis
            patient_info: Dictionary of patient information
            
        Returns:
            dict: Dictionary with report sections
        """
        report = {}
        
        for section in config.REPORT_SECTIONS:
            report[section] = self.generate_report_section(section, findings, patient_info)
        
        return report 