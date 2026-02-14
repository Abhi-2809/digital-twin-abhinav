"""LangChain tools for Google Calendar integration and daily news agent."""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from src.tools.google_clients.calendar_client import list_events, create_event, delete_event
from src.tools.google_clients.email_client import send_email

# Default timezone for calendar: all user times without timezone are interpreted as PST
PST = ZoneInfo("America/Los_Angeles")


def _parse_to_utc(iso_time: str) -> datetime:
    """Parse ISO 8601 time string; if naive, treat as PST and return UTC."""
    s = iso_time.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=PST).astimezone(ZoneInfo("UTC"))
    else:
        dt = dt.astimezone(ZoneInfo("UTC"))
    return dt


@tool
def get_calendar_events(start_time: str, end_time: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Get events from my Google Calendar between start_time and end_time.

    Use this tool when the user asks about their schedule, plans, meetings, or events
    for a specific time period. Examples:
    - "What are you doing this weekend?"
    - "What is your plan for this week?"
    - "Show me my schedule for next Monday"

    Times: Use ISO 8601 (e.g. 2026-02-15T09:00:00 or 2026-02-15T09:00:00-08:00).
    If no timezone is given, the time is interpreted as PST (America/Los_Angeles).
    
    Args:
        start_time: Start time in ISO 8601 format (e.g., "2026-02-15T00:00:00" or with -08:00 for PST)
        end_time: End time in ISO 8601 format (e.g., "2026-02-16T23:59:59")
        max_results: Maximum number of events to return (default: 10)
        
    Returns:
        List of event dictionaries with summary, start, end, description, location, attendees
    """
    try:
        start_dt = _parse_to_utc(start_time)
        end_dt = _parse_to_utc(end_time)
        events = list_events(start_dt, end_dt, max_results)
        return events
    except ValueError as e:
        return [{"error": f"Invalid date format: {str(e)}. Use ISO 8601 (e.g. 2026-02-15T09:00:00 or 2026-02-15T09:00:00-08:00)."}]
    except Exception as e:
        return [{"error": f"Failed to retrieve calendar events: {str(e)}"}]


@tool
def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a new event in my Google Calendar.

    Use this tool when the user wants to schedule, book, or create a meeting/event.
    Examples: "Schedule a meeting on Friday at 3pm", "Add event tomorrow at 9am for 1 hour".

    Times: Use ISO 8601 (e.g. 2026-02-15T09:00:00 or 2026-02-15T09:00:00-08:00).
    If no timezone is given, the time is interpreted as PST (America/Los_Angeles).
    So "9am" should be passed as 09:00 in local date (PST), e.g. 2026-02-15T09:00:00.
    
    Args:
        summary: Event title/name (required)
        start_time: Start time in ISO 8601 format (naive or with -08:00 for PST)
        end_time: End time in ISO 8601 format
        description: Optional event description
        location: Optional event location
        attendees: Optional list of attendee email addresses
        
    Returns:
        Dictionary with created event details including id, summary, start, end, htmlLink
    """
    try:
        start_dt = _parse_to_utc(start_time)
        end_dt = _parse_to_utc(end_time)
        if end_dt <= start_dt:
            return {"error": "End time must be after start time"}
        created_event = create_event(
            summary=summary,
            start_datetime=start_dt,
            end_datetime=end_dt,
            description=description,
            location=location,
            attendees=attendees
        )
        # Include a clear message with the event link so the agent can show it to the user
        link = created_event.get("htmlLink", "")
        if link:
            created_event["message"] = f"Event created. View in Google Calendar: {link}"
        return created_event
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}. Use ISO 8601; no timezone = PST (e.g. 2026-02-15T09:00:00)."}
    except Exception as e:
        return {"error": f"Failed to create calendar event: {str(e)}"}


@tool
def delete_calendar_event(event_id: str) -> Dict[str, Any]:
    """Delete an event from my Google Calendar by its event ID.

    Use this tool when the user wants to cancel or delete a calendar event.
    You do NOT have the event_id from the user's message. To delete an event the user
    describes by time or name (e.g. "delete breakfast tomorrow 6am"):
    1. First call get_calendar_events(start_time, end_time) for that time range.
    2. From the returned list, find the event whose "summary" matches (e.g. "breakfast").
    3. Call this tool with that event's "id" field as event_id.
    Only after delete_calendar_event returns success should you say the event was deleted.

    Args:
        event_id: The calendar event ID (required). Get it from get_calendar_events response "id" field.

    Returns:
        Dictionary with deletion status and event_id
    """
    try:
        if not event_id or not event_id.strip():
            return {"error": "Event ID is required to delete an event"}
        
        result = 
        (event_id.strip())
        return result
    
    except Exception as e:
        return {"error": f"Failed to delete calendar event: {str(e)}"}


@tool
def send_email_to_abhinav(subject: str, body: str) -> Dict[str, Any]:
    """Send an email to Abhinav (arabellyabhinav2809@gmail.com).
    
    Use this tool when the user asks to send an email to Abhinav.
    Examples:
    - "Send an email to Abhinav with subject 'Meeting reminder' and body 'Don't forget about our meeting tomorrow'"
    - "Email Abhinav about the project update"
    - "Send a message to Abhinav with subject X and body Y"
    
    Args:
        subject: Email subject line (required)
        body: Email body/content (required)
        
    Returns:
        Dictionary with success status, message_id, and email details
    """
    try:
        if not subject or not subject.strip():
            return {"error": "Email subject is required"}
        if not body or not body.strip():
            return {"error": "Email body is required"}
        
        result = send_email(
            to='arabellyabhinav2809@gmail.com',
            subject=subject.strip(),
            body=body.strip()
        )
        return result
    
    except Exception as e:
        return {"error": f"Failed to send email: {str(e)}"}


# Three topics, three well-known public RSS feeds (no API key, free, stable)
NEWS_FEEDS = {
    "technology": {
        "label": "TECHNOLOGY",
        "domain": "TechCrunch",
        "rss": "https://techcrunch.com/feed/",
    },
    "world": {
        "label": "WORLD NEWS",
        "domain": "BBC News",
        "rss": "https://feeds.bbci.co.uk/news/rss.xml",
    },
    "sports": {
        "label": "SPORTS",
        "domain": "BBC Sport",
        "rss": "https://feeds.bbci.co.uk/sport/rss.xml",
    },
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _fetch_rss_articles(rss_url: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Fetch latest articles from an RSS or Atom feed. Supports both <item> (RSS) and <entry> (Atom)."""
    articles = []
    try:
        resp = requests.get(rss_url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        try:
            soup = BeautifulSoup(resp.text, "xml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")
        if not soup:
            return articles

        # RSS 2.0: <item> with <title>, <link>, <description>
        items = soup.find_all("item")
        if not items:
            # Atom: <entry> with <title>, <link href="...">, <summary> or <content>
            items = soup.find_all("entry")

        for item in items:
            if len(articles) >= num_results:
                break
            title_tag = item.find("title")
            link_tag = item.find("link")
            desc_tag = item.find("description") or item.find("summary") or item.find("content")

            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            # Link: RSS often has link text; Atom has <link href="..."/>
            url = ""
            if link_tag:
                url = link_tag.get("href", "").strip()
                if not url:
                    url = link_tag.get_text(strip=True)
                if not url and link_tag.next_sibling:
                    url = str(link_tag.next_sibling).strip()
            if not url:
                continue
            snippet = ""
            if desc_tag:
                snippet = desc_tag.get_text(strip=True)[:500]
            articles.append({"title": title, "url": url, "snippet": snippet})
    except requests.RequestException as e:
        logging.getLogger(__name__).warning("RSS fetch failed for %s: %s", rss_url, e)
    except Exception as e:
        logging.getLogger(__name__).warning("RSS parse failed for %s: %s", rss_url, e)
    return articles


def _fetch_article_text(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract main text content from an article URL."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text[:max_chars]
    except Exception:
        return ""


def _summarize_article(title: str, text: str) -> str:
    """Summarize an article to 75 words max using the LLM."""
    if not text:
        return "Could not fetch article content for summarization."
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=120)
        prompt = (
            f"Summarize the following article in 75 words or fewer. "
            f"Be concise and factual.\n\nTitle: {title}\n\nContent: {text}"
        )
        response = llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        return f"Summary unavailable ({str(e)})"


@tool
def get_daily_news_summary(query: str = "") -> str:
    """Get a daily news summary from TechCrunch (technology), BBC News (world), and BBC Sport (sports).

    Use this tool when the user asks for a news update, daily brief, today's summary,
    or news summary. Examples:
    - "Give me today's summary"
    - "What's the news update?"
    - "Daily brief"
    - "News summary"

    Args:
        query: The user's query (used for trigger detection, can be empty)

    Returns:
        A formatted Markdown news digest grouped by topic with article summaries and links
    """
    today_display = datetime.now().strftime("%b %d, %Y")
    sections = []

    for topic, feed_info in NEWS_FEEDS.items():
        topic_label = feed_info["label"]
        domain = feed_info["domain"]
        articles = _fetch_rss_articles(feed_info["rss"], num_results=3)

        if not articles:
            sections.append(f"### {topic_label} ({domain})\nNo recent articles found today.\n")
            continue

        article_lines = []
        for art in articles:
            text = _fetch_article_text(art["url"])
            summary = _summarize_article(art["title"], text)
            # Title as markdown link so each summary is clearly linked to the article
            article_lines.append(
                f"- **[{art['title']}]({art['url']})**\n  {summary}\n"
            )

        sections.append(
            f"### {topic_label} ({domain})\n" + "\n".join(article_lines)
        )

    digest = f"## TODAY'S NEWS SUMMARY - {today_display}\n\n" + "\n".join(sections)
    return digest


def get_all_tools() -> List:
    """Get all tools (calendar, email, and news) for LangChain/LangGraph integration.

    Returns:
        List of all tool instances
    """
    return [
        get_calendar_events,
        create_calendar_event,
        delete_calendar_event,
        send_email_to_abhinav,
        get_daily_news_summary,
    ]


def get_calendar_tools() -> List:
    """Get all tools for LangChain/LangGraph integration.

    Returns:
        List of tool instances (calendar, email, and news tools)
    """
    return get_all_tools()
