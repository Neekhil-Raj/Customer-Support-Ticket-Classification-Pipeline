import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

def generate_sample_data(n_samples=1000):
    """Generate sample customer support ticket data"""
    
    # Sample data pools
    customer_names = ["John Smith", "Sarah Johnson", "Mike Davis", "Emily Brown", "David Wilson", 
                     "Lisa Anderson", "Chris Martin", "Jennifer Garcia", "Robert Taylor", "Amanda White"]
    
    products = ["Software Pro", "Basic Plan", "Premium Suite", "Enterprise Edition", "Mobile App"]
    
    ticket_types = ["Technical", "Billing", "Authentication", "General Inquiry", "Feature Request"]
    
    channels = ["Email", "Chat", "Phone", "Web"]
    
    priorities = ["Low", "Medium", "High", "Critical"]
    
    statuses = ["Open", "In Progress", "Resolved", "Closed"]
    
    # Technical issues
    technical_subjects = [
        "Software crashes on startup",
        "Cannot save files",
        "Application freezes",
        "Export function not working",
        "Performance issues",
        "Installation failed",
        "Database connection error"
    ]
    
    technical_descriptions = [
        "The application crashes every time I try to start it. Error message shows memory allocation failed.",
        "When I try to save my document, nothing happens and I lose all my work.",
        "The software becomes unresponsive after 10 minutes of use.",
        "Export to PDF feature returns an empty file.",
        "The application is very slow and takes forever to load.",
        "Installation process stops at 80% and shows unknown error.",
        "Cannot connect to the database, getting timeout errors."
    ]
    
    # Billing issues
    billing_subjects = [
        "Billing inquiry",
        "Refund request",
        "Payment failed",
        "Subscription cancellation",
        "Invoice discrepancy"
    ]
    
    billing_descriptions = [
        "I was charged twice for my monthly subscription. Please help me get a refund.",
        "I want to cancel my subscription immediately and get a prorated refund.",
        "My payment method was declined but I have sufficient funds.",
        "The invoice amount doesn't match what I agreed to pay.",
        "I need clarification on the charges in my latest bill."
    ]
    
    # Authentication issues  
    auth_subjects = [
        "Cannot login",
        "Password reset not working",
        "Account locked",
        "Two-factor authentication issues"
    ]
    
    auth_descriptions = [
        "I forgot my password and the reset email is not arriving.",
        "My account is locked after failed login attempts.",
        "Two-factor authentication code is not working.",
        "Cannot access my account with correct credentials."
    ]
    
    # General inquiries
    general_subjects = [
        "How to use feature X",
        "Training request",
        "Documentation request",
        "General question"
    ]
    
    general_descriptions = [
        "I need help understanding how to use the reporting feature.",
        "Can you provide training materials for new users?",
        "Where can I find the user manual?",
        "What are the system requirements for your software?"
    ]
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        ticket_type = random.choice(ticket_types)
        
        # Select subject and description based on ticket type
        if ticket_type == "Technical":
            subject = random.choice(technical_subjects)
            description = random.choice(technical_descriptions)
        elif ticket_type == "Billing":
            subject = random.choice(billing_subjects)
            description = random.choice(billing_descriptions)
        elif ticket_type == "Authentication":
            subject = random.choice(auth_subjects)
            description = random.choice(auth_descriptions)
        else:
            subject = random.choice(general_subjects)
            description = random.choice(general_descriptions)
        
        # Generate random dates
        purchase_date = datetime.now() - timedelta(days=random.randint(1, 365))
        response_time = timedelta(minutes=random.randint(15, 480))
        resolution_time = random.randint(1, 168)  # hours
        
        record = {
            'ticket_id': f"TKT-{str(uuid.uuid4())[:8].upper()}",
            'customer_name': random.choice(customer_names),
            'customer_email': f"{random.choice(customer_names).lower().replace(' ', '.')}@email.com",
            'customer_age': random.randint(18, 70),
            'customer_gender': random.choice(['Male', 'Female', 'Other']),
            'product_purchased': random.choice(products),
            'date_of_purchase': purchase_date.strftime('%Y-%m-%d'),
            'ticket_type': ticket_type,
            'ticket_subject': subject,
            'ticket_description': description,
            'ticket_status': random.choice(statuses),
            'resolution': "Issue resolved" if random.choice([True, False]) else "",
            'ticket_priority': random.choice(priorities),
            'ticket_channel': random.choice(channels),
            'first_response_time': response_time.total_seconds() / 3600,  # hours
            'time_to_resolution': resolution_time,
            'customer_satisfaction_rating': round(random.uniform(1.0, 5.0), 1)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(1000)
    
    # Save to CSV
    df.to_csv('data/sample_data.csv', index=False)
    print(f"Generated {len(df)} sample tickets and saved to data/sample_data.csv")
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Shape: {df.shape}")
    print(f"\nTicket Types Distribution:")
    print(df['ticket_type'].value_counts())
    print(f"\nPriority Distribution:")
    print(df['ticket_priority'].value_counts())
