"""Gmail API client for sending emails."""

import os
import base64
from email.mime.text import MIMEText
from typing import Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()

# Gmail API scopes
# Note: Combined with Calendar scopes - both APIs use the same token file
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.send'
]

# Token file path (shared with calendar)
TOKEN_FILE = 'token.json'


def get_credentials() -> Credentials:
    """Get valid user credentials from storage or OAuth flow.
    
    Returns:
        Credentials object for Gmail API (with combined Calendar + Gmail scopes)
    """
    creds = None
    
    # Load existing token if available
    # Try to load with combined scopes
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except ValueError:
            # If token doesn't have Gmail scope, we'll need to re-authenticate
            creds = None
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Get credentials path from environment variable
            creds_path = os.getenv('GOOGLE_CALENDAR_CREDENTIALS_PATH')
            if not creds_path:
                raise ValueError(
                    "GOOGLE_CALENDAR_CREDENTIALS_PATH not found in environment variables. "
                    "Please set it to the path of your Google OAuth credentials JSON file."
                )
            
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Credentials file not found at: {creds_path}. "
                    "Please ensure the path is correct."
                )
            
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return creds


def get_gmail_service():
    """Get Gmail service instance.
    
    Returns:
        Gmail API service object
    """
    creds = get_credentials()
    return build('gmail', 'v1', credentials=creds)


def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Send an email using Gmail API.
    
    Args:
        to: Recipient email address (will always send to arabellyabhinav2809@gmail.com)
        subject: Email subject
        body: Email body/content
        
    Returns:
        Dictionary with success status and message ID
    """
    try:
        service = get_gmail_service()
        
        # Always send to arabellyabhinav2809@gmail.com
        recipient = 'arabellyabhinav2809@gmail.com'
        
        # Create message
        message = MIMEText(body)
        message['to'] = recipient
        message['subject'] = subject
        
        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # Send message
        send_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        
        return {
            'success': True,
            'message_id': send_message.get('id', ''),
            'to': recipient,
            'subject': subject,
            'status': 'sent'
        }
    
    except HttpError as error:
        raise Exception(f"An error occurred while sending email: {error}")
    except Exception as e:
        raise Exception(f"Failed to send email: {str(e)}")
