"""Google Calendar API client for Abhinav Digital Twin."""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()

# Google Calendar API scopes
# Combined with Gmail scope - both APIs use the same token file
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.send'
]

# Token file path
TOKEN_FILE = 'token.json'


def get_credentials() -> Credentials:
    """Get valid user credentials from storage or OAuth flow.
    
    Returns:
        Credentials object for Google Calendar API
    """
    creds = None
    
    # Load existing token if available
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
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


def get_calendar_service():
    """Get Google Calendar service instance.
    
    Returns:
        Google Calendar API service object
    """
    creds = get_credentials()
    return build('calendar', 'v3', credentials=creds)


def list_events(
    start_datetime: datetime,
    end_datetime: datetime,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """List events from Google Calendar within a time range.
    
    Args:
        start_datetime: Start datetime for event query
        end_datetime: End datetime for event query
        max_results: Maximum number of events to return
        
    Returns:
        List of event dictionaries with summary, start, end, description, location, etc.
    """
    try:
        service = get_calendar_service()
        
        # Convert to RFC3339 for API. Tool layer passes UTC datetimes.
        time_min = (start_datetime.isoformat() + "Z") if start_datetime.tzinfo is None else start_datetime.isoformat().replace("+00:00", "Z")
        time_max = (end_datetime.isoformat() + "Z") if end_datetime.tzinfo is None else end_datetime.isoformat().replace("+00:00", "Z")
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Format events for easier consumption
        formatted_events = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            formatted_events.append({
                'id': event.get('id', ''),
                'summary': event.get('summary', 'No Title'),
                'start': start,
                'end': end,
                'description': event.get('description', ''),
                'location': event.get('location', ''),
                'attendees': [att.get('email', '') for att in event.get('attendees', [])],
                'status': event.get('status', 'confirmed')
            })
        
        return formatted_events
    
    except HttpError as error:
        raise Exception(f"An error occurred while fetching calendar events: {error}")
    except Exception as e:
        raise Exception(f"Failed to list calendar events: {str(e)}")


def create_event(
    summary: str,
    start_datetime: datetime,
    end_datetime: datetime,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a new event in Google Calendar.
    
    Args:
        summary: Event title/summary
        start_datetime: Event start datetime
        end_datetime: Event end datetime
        description: Optional event description
        location: Optional event location
        attendees: Optional list of attendee email addresses
        
    Returns:
        Dictionary with created event details
    """
    try:
        service = get_calendar_service()
        
        # Build event body (UTC; tool layer converts user PST to UTC)
        start_str = (start_datetime.isoformat() + "Z") if start_datetime.tzinfo is None else start_datetime.isoformat().replace("+00:00", "Z")
        end_str = (end_datetime.isoformat() + "Z") if end_datetime.tzinfo is None else end_datetime.isoformat().replace("+00:00", "Z")
        event_body = {
            "summary": summary,
            "start": {"dateTime": start_str, "timeZone": "UTC"},
            "end": {"dateTime": end_str, "timeZone": "UTC"},
        }
        
        if description:
            event_body['description'] = description
        
        if location:
            event_body['location'] = location
        
        if attendees:
            event_body['attendees'] = [{'email': email} for email in attendees]
        
        # Create the event
        created_event = service.events().insert(
            calendarId='primary',
            body=event_body
        ).execute()
        
        return {
            'id': created_event.get('id', ''),
            'summary': created_event.get('summary', ''),
            'start': created_event['start'].get('dateTime', created_event['start'].get('date')),
            'end': created_event['end'].get('dateTime', created_event['end'].get('date')),
            'description': created_event.get('description', ''),
            'location': created_event.get('location', ''),
            'attendees': [att.get('email', '') for att in created_event.get('attendees', [])],
            'htmlLink': created_event.get('htmlLink', ''),
            'status': 'created'
        }
    
    except HttpError as error:
        raise Exception(f"An error occurred while creating calendar event: {error}")
    except Exception as e:
        raise Exception(f"Failed to create calendar event: {str(e)}")


def find_available_slots(
    start_datetime: datetime,
    end_datetime: datetime,
    duration_minutes: int = 60
) -> Dict[str, Any]:
    """Find available time slots within a date range.
    
    Args:
        start_datetime: Start datetime for availability check
        end_datetime: End datetime for availability check
        duration_minutes: Duration of slot to check for (default: 60 minutes)
        
    Returns:
        Dictionary with busy times and available slots
    """
    try:
        # Get all events in the time range
        events = list_events(start_datetime, end_datetime, max_results=100)
        
        # Parse busy times
        busy_times = []
        for event in events:
            try:
                event_start = datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                event_end = datetime.fromisoformat(event['end'].replace('Z', '+00:00'))
                busy_times.append((event_start, event_end))
            except (ValueError, KeyError):
                continue
        
        # Sort busy times
        busy_times.sort(key=lambda x: x[0])
        
        # Find available slots
        available_slots = []
        current_time = start_datetime
        
        for busy_start, busy_end in busy_times:
            # Check if there's a gap before this busy period
            if current_time < busy_start:
                gap_duration = (busy_start - current_time).total_seconds() / 60
                if gap_duration >= duration_minutes:
                    available_slots.append({
                        'start': current_time.isoformat(),
                        'end': busy_start.isoformat(),
                        'duration_minutes': int(gap_duration)
                    })
            # Move current_time to end of busy period
            if busy_end > current_time:
                current_time = busy_end
        
        # Check if there's time available after the last busy period
        if current_time < end_datetime:
            gap_duration = (end_datetime - current_time).total_seconds() / 60
            if gap_duration >= duration_minutes:
                available_slots.append({
                    'start': current_time.isoformat(),
                    'end': end_datetime.isoformat(),
                    'duration_minutes': int(gap_duration)
                })
        
        return {
            'busy_times': [
                {'start': start.isoformat(), 'end': end.isoformat()}
                for start, end in busy_times
            ],
            'available_slots': available_slots,
            'total_busy_minutes': sum(
                (end - start).total_seconds() / 60
                for start, end in busy_times
            ),
            'total_available_minutes': sum(slot['duration_minutes'] for slot in available_slots)
        }
    
    except Exception as e:
        raise Exception(f"Failed to find available slots: {str(e)}")


def delete_event(event_id: str) -> Dict[str, Any]:
    """Delete an event from Google Calendar.
    
    Args:
        event_id: ID of the event to delete
        
    Returns:
        Dictionary with deletion status
    """
    try:
        service = get_calendar_service()
        
        # Delete the event
        service.events().delete(
            calendarId='primary',
            eventId=event_id
        ).execute()
        
        return {
            'success': True,
            'event_id': event_id,
            'status': 'deleted',
            'message': f'Event {event_id} successfully deleted'
        }
    
    except HttpError as error:
        if error.resp.status == 404:
            raise Exception(f"Event not found: {event_id}")
        raise Exception(f"An error occurred while deleting calendar event: {error}")
    except Exception as e:
        raise Exception(f"Failed to delete calendar event: {str(e)}")
