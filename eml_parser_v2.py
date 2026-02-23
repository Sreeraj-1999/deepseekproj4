"""
EML Mail Parser v2 - Per-Thread-Email Processing
==================================================
Key changes from v1:
1. Processes each email in thread INDIVIDUALLY (chronological order)
2. Product-stage association - only updates products mentioned in THAT email
3. Extracts sender/date per thread email
4. Monotonic state - stages only move forward
5. Better context-aware stage detection
6. Proper remarks per email

Usage:
    python eml_parser_v2.py "C:/path/to/email.eml"
    python eml_parser_v2.py "C:/path/to/email_folder/"
"""

import email
import email.policy
import os
import sys
import re
import json
import csv
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import List, Dict, Optional, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

KNOWN_VESSELS = [
    "molly schulte",
    "clemens schulte",
    "maren schulte",
    "anna schulte",
    # Add more here
]

PRODUCTS = ["MDC", "Transbox", "Electrical Panel", "Flowmeter", "Energy Meter", "SPM"]

# Product detection patterns - more precise than v1
PRODUCT_PATTERNS = {
    "MDC": [r'\bMDC\b', r'\bMDC\s+PC\b', r'\bdata\s+collector\b'],
    "Transbox": [r'\btransbox\b', r'\btrans[\s-]?box\b'],
    "Electrical Panel": [
        r'(?:MariApps|Memphis|mariapps)\s+(?:electrical\s+)?panel',
        r'electrical\s+panel',
    ],
    "Flowmeter": [r'\bflow\s*meter\b', r'\bF050\b', r'\bF025\b', r'\bPromass\b'],
    "Energy Meter": [r'\benergy\s+meter\b', r'\bRogowski\s+coil', r'\benergy\s+meters\b'],
    "SPM": [r'\bSPM\b', r'\bshaft\s+power\b'],
}

# Stage detection - context-aware patterns
# Each pattern: (regex, stage, status, description)
STAGE_RULES = [
    # === PROCUREMENT ===
    (r'(?:parts|equipment|hardware|items|cargo|goods)\s+(?:arrived|received|on\s+board|delivered)', 
     'procurement', 'done', 'Items received on board'),
    (r'vessel\s+has\s+not\s+yet\s+received', 
     'procurement', 'pending', 'Items not yet received'),
    (r'not\s+(?:yet\s+)?(?:received|supplied|delivered)',
     'procurement', 'pending', 'Items not yet received'),
    (r'(?:will\s+be\s+)?(?:supplied|dispatched|shipped)\s+(?:later|soon|separately)',
     'procurement', 'in_progress', 'Items being arranged'),
    (r'(?:ordered|purchase\s+order|PO\s+raised)',
     'procurement', 'in_progress', 'Order placed'),
    (r'(?:we\s+will\s+supply|will\s+share|will\s+be\s+supplied)',
     'procurement', 'in_progress', 'Supply planned'),
    (r'not\s+supplied',
     'procurement', 'pending', 'Not yet supplied'),

    # === INSTALL ===
    # Done indicators
    (r'installation\s+(?:of\s+.{3,40}\s+)?(?:completed|done|finished)',
     'install', 'done', 'Installation completed'),
    (r'(?:has|have)\s+been\s+(?:completed|installed|mounted|fitted)',
     'install', 'done', 'Installation completed'),
    (r'(?:installation|cable\s+laying|LAN\s+cable|termination).*(?:completed|done)',
     'install', 'done', 'Installation work completed'),
    (r'panel\s+installation\s+completed',
     'install', 'done', 'Panel installed'),
    (r'(?:connected|installed|mounted|fitted)\s+(?:to|on|at|in)\s+(?:port|rack|location|panel|switch)',
     'install', 'done', 'Hardware connected'),
    (r'connections?\s+were\s+made',
     'install', 'done', 'Connections completed'),
    (r'following\s+jobs?\s+has\s+been\s+completed',
     'install', 'done', 'Jobs completed'),
    (r'LAN\s+cable[s]?\s+.*terminated',
     'install', 'done', 'Cables terminated'),
    
    # In-progress indicators
    (r'(?:we\s+will|will\s+)?(?:proceed|carry\s+out|start)\s+(?:with\s+)?(?:the\s+)?(?:install|cable\s+lay|LAN)',
     'install', 'in_progress', 'Installation starting'),
    (r'(?:started|commenced|begun)\s+.*(?:install|connection|cable|wiring)',
     'install', 'in_progress', 'Installation in progress'),
    (r'(?:please\s+)?(?:proceed|install|connect|mount)\s+(?:the|with)',
     'install', 'in_progress', 'Installation instructions given'),
    (r'we\s+have\s+(?:started|identified|proposed)',
     'install', 'in_progress', 'Installation planning'),
    
    # Pending
    (r'(?:yet\s+to|not\s+yet)\s+install',
     'install', 'pending', 'Not yet installed'),
    (r'(?:awaiting|pending)\s+installation',
     'install', 'pending', 'Installation pending'),

    # === COMMISSION ===
    # Done indicators
    (r'(?:commissioned|commissioning\s+completed|system\s+(?:online|operational))',
     'commission', 'done', 'Commissioning completed'),
    (r'(?:remote\s+)?access\s+(?:working|established|successful)',
     'commission', 'done', 'Remote access working'),
    (r'data\s+(?:flowing|receiving|coming)',
     'commission', 'done', 'Data flow established'),
    (r'telemetry\s+(?:active|working|online)',
     'commission', 'done', 'Telemetry active'),
    (r'successfully\s+(?:configured|connected|accessed)',
     'commission', 'done', 'Configuration successful'),

    # In-progress indicators
    (r'unable\s+to\s+access',
     'commission', 'in_progress', 'Access issues'),
    (r'(?:cannot|can\'t|could\s+not)\s+(?:access|connect|reach)',
     'commission', 'in_progress', 'Connection issues'),
    (r'(?:network|connectivity|access)\s+issue',
     'commission', 'in_progress', 'Network issue'),
    (r'(?:try|trying)\s+(?:to\s+)?access',
     'commission', 'in_progress', 'Attempting access'),
    (r'(?:not\s+been\s+)?powered\s+(?:ON|on)',
     'commission', 'in_progress', 'Power issue being resolved'),
    (r'(?:troubleshoot|configur|test)',
     'commission', 'in_progress', 'Troubleshooting'),
    (r'(?:please|kindly)\s+(?:try|check|verify|confirm)\s+(?:access|connect)',
     'commission', 'in_progress', 'Verification requested'),
    (r'share\s+(?:a\s+)?screenshot',
     'commission', 'in_progress', 'Diagnostic info requested'),
    (r'(?:log\s+in|login|credentials|username|password)',
     'commission', 'in_progress', 'Login/access being configured'),

    # Pending
    (r'(?:yet\s+to|not\s+yet)\s+commission',
     'commission', 'pending', 'Not yet commissioned'),

    # === FAT ===
    (r'FAT\s+(?:completed|done|passed|signed)',
     'fat', 'done', 'FAT completed'),
    (r'(?:acceptance\s+test|FAT)\s+(?:in\s+progress|scheduled|ongoing)',
     'fat', 'in_progress', 'FAT in progress'),
    (r'(?:FAT|acceptance\s+test)\s+(?:pending|not\s+done)',
     'fat', 'pending', 'FAT pending'),
]

# Blockers / issues to capture
BLOCKER_PATTERNS = [
    (r'no\s+space\s+(?:on|in)\s+(?:the\s+)?rail', 'No space on rail for installation'),
    (r'not\s+(?:yet\s+)?received\s+(?:telemetry\s+)?switch', 'Telemetry switch not received'),
    (r'network\s+issue', 'Network connectivity issue'),
    (r'unable\s+to\s+access', 'Remote access issue'),
    (r'not\s+(?:been\s+)?powered\s+(?:on|ON)', 'Equipment not powered on'),
    (r'awaiting\s+(?:clarification|confirmation|approval)', 'Awaiting clarification'),
    (r'(?:require|need|required)\s+(?:clarification|additional|further)', 'Clarification needed'),
]

# Signature/noise patterns
NOISE_PATTERNS = [
    r"Thanks\s*[&and]*\s*Regards.*",
    r"Sincerely,.*",
    r"Best\s+[Rr]egards.*",
    r"Kind\s+[Rr]egards.*",
    r"MariApps\s+Marine\s+Solutions.*",
    r"MariApps\s+House.*",
    r"www\.mariapps\.com.*",
    r"www\.memphis-marine\.com.*",
    r"Tel(?:ephone)?:.*",
    r"Telex:.*",
    r"FBB\s+Phone:.*",
    r"IP\s+Phone.*",
    r"Email:\s*master@.*",
    r"\|.*(?:Engineer|Manager|Director|Officer).*@.*",
    r"Plot\s+No\s+A2.*",
    r"SmartCity\s+Kochi.*",
    r"Kerala\s*[–-]?\s*\d+",
    r"Mob:\s*\+?\d+",
    r"\[.*?Logo.*?\]",
    r"\[.*?Description\s+automatically\s+generated.*?\]",
    r"CLICK\s+HERE\s+FOR.*",
    r"_{3,}",
    r"<mailto:.*?>",
    r"<https?://.*?>",
    r"mailto:\S+",
]


# =============================================================================
# THREAD EMAIL PARSER
# =============================================================================

class ThreadEmail:
    """Represents a single email within a thread."""
    
    def __init__(self, raw_text: str, index: int):
        self.raw_text = raw_text
        self.index = index
        self.sender = None
        self.sender_email = None
        self.sender_role = None  # 'ship' or 'office'
        self.date = None
        self.date_str = None
        self.clean_body = None
        self.products_mentioned = []
        self.stage_updates = {}
        self.blockers = []
        self.remarks = []
        
        self._parse()
    
    def _parse(self):
        """Parse all fields from raw text."""
        self._extract_sender()
        self._extract_date()
        self._clean_body_text()
        self._extract_products()
        self._extract_stages()
        self._extract_blockers()
        self._extract_remarks()
    
    def _extract_sender(self):
        """Extract sender name and classify as ship or office."""
        # Look for From: header in forwarded emails
        from_match = re.search(
            r'From:\s*(?:"?([^"<\r\n]+)"?\s*)?<?([^>\r\n]+@[^>\r\n]+)>?',
            self.raw_text[:500]
        )
        
        if from_match:
            self.sender = (from_match.group(1) or '').strip().strip('"')
            self.sender_email = (from_match.group(2) or '').strip()
        
        # Classify sender
        text = self.raw_text
        if self.sender_email and 'bsmfleet.com' in self.sender_email:
            self.sender_role = 'ship'
        elif self.sender_email and ('mariapps.com' in self.sender_email or 'memphis-marine.com' in self.sender_email):
            self.sender_role = 'office'
        
        # Also check for CE / Captain signatures
        if re.search(r'(?:C/E|Chief\s+Engineer)\b', text):
            self.sender_role = 'ship'
        
        # Extract CE name
        ce_match = re.search(r'(?:CE|C/E)[\s.]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        if ce_match and not self.sender:
            self.sender = f"CE {ce_match.group(1)}"
            self.sender_role = 'ship'
    
    def _extract_date(self):
        """Extract date from the email header within thread."""
        # Pattern: Sent: Friday, November 14, 2025 9:19 PM
        sent_match = re.search(
            r'Sent:\s*(?:\w+,\s*)?(\w+\s+\d{1,2},?\s+\d{4}(?:\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)?)',
            self.raw_text[:600]
        )
        if sent_match:
            self.date_str = sent_match.group(1).strip()
            self._parse_date(self.date_str)
            return
        
        # Pattern: Sent: 14 November 2025 00:15
        sent_match2 = re.search(
            r'Sent:\s*(\d{1,2}\s+\w+\s+\d{4}(?:\s+\d{1,2}:\d{2})?)',
            self.raw_text[:600]
        )
        if sent_match2:
            self.date_str = sent_match2.group(1).strip()
            self._parse_date(self.date_str)
    
    def _parse_date(self, date_str: str):
        """Parse various date formats."""
        formats = [
            '%B %d, %Y %I:%M %p',
            '%B %d, %Y',
            '%B %d %Y %I:%M %p',
            '%B %d %Y',
            '%d %B %Y %H:%M',
            '%d %B %Y',
            '%b %d, %Y %I:%M %p',
            '%b %d, %Y',
            '%d %b %Y %H:%M',
            '%d %b %Y',
        ]
        
        clean = re.sub(r'\s+', ' ', date_str).strip().rstrip(',')
        
        for fmt in formats:
            try:
                self.date = datetime.strptime(clean, fmt)
                return
            except ValueError:
                continue
    
    def _clean_body_text(self):
        """Remove headers, signatures, noise from body."""
        text = self.raw_text
        
        # Remove From/To/Cc/Subject headers
        text = re.sub(
            r'^From:.*?(?=\n\n|\r\n\r\n)',
            '', text, count=1, flags=re.DOTALL
        )
        # Also remove individual header lines that might remain
        text = re.sub(r'^(?:To|Cc|Subject|Sent):.*$', '', text, flags=re.MULTILINE)
        
        # Remove noise patterns
        for pattern in NOISE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Clean whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        self.clean_body = text.strip()
    
    def _extract_products(self):
        """Detect which products are mentioned in THIS specific email."""
        found = []
        text = self.raw_text  # Search in full text including headers for context
        
        for product, patterns in PRODUCT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found.append(product)
                    break
        
        self.products_mentioned = found
    
    def _extract_stages(self):
        """Extract stage updates from this email's content."""
        text = self.raw_text
        
        for pattern, stage, status, description in STAGE_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                if stage not in self.stage_updates:
                    self.stage_updates[stage] = {
                        'status': status,
                        'description': description,
                        'matched_pattern': pattern[:50]
                    }
                else:
                    # Keep higher priority status
                    current_priority = _status_priority(self.stage_updates[stage]['status'])
                    new_priority = _status_priority(status)
                    if new_priority > current_priority:
                        self.stage_updates[stage] = {
                            'status': status,
                            'description': description,
                            'matched_pattern': pattern[:50]
                        }
    
    def _extract_blockers(self):
        """Extract blockers/issues mentioned."""
        text = self.raw_text
        
        for pattern, description in BLOCKER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                self.blockers.append(description)
    
    def _extract_remarks(self):
        """Generate concise remarks from email content."""
        if not self.clean_body:
            return
        
        lines = [l.strip() for l in self.clean_body.split('\n') if l.strip()]
        
        for line in lines:
            # Skip greetings and noise
            if re.match(r'^(Dear|Good\s+day|Hi|Hello|Noted\s+with\s+thanks)', line, re.IGNORECASE):
                continue
            if len(line) < 15 or len(line) > 200:
                continue
            if re.match(r'^(From:|To:|Cc:|Subject:|Sent:|\*)', line, re.IGNORECASE):
                continue
            if re.match(r'^\d+\.\s*$', line):
                continue
            
            self.remarks.append(line.strip())
            
            if len(self.remarks) >= 3:
                break
    
    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'sender': self.sender,
            'sender_email': self.sender_email,
            'sender_role': self.sender_role,
            'date': self.date.isoformat() if self.date else None,
            'date_str': self.date_str,
            'products_mentioned': self.products_mentioned,
            'stage_updates': self.stage_updates,
            'blockers': self.blockers,
            'remarks': self.remarks,
            'clean_body_preview': (self.clean_body or '')[:200],
        }


def _status_priority(status: str) -> int:
    return {"pending": 0, "in_progress": 1, "done": 2}.get(status, -1)


# =============================================================================
# EML PARSER v2
# =============================================================================

class EMLParserV2:
    """Parse EML files with per-thread-email processing."""
    
    def __init__(self):
        self.noise_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NOISE_PATTERNS]
    
    def parse_eml(self, file_path: str) -> Dict:
        """Parse a single EML file into structured data."""
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        # Extract top-level headers
        parsed = {
            'file': str(file_path.name),
            'subject': str(msg['subject'] or ''),
            'from': str(msg['from'] or ''),
            'to': str(msg['to'] or ''),
            'cc': str(msg['cc'] or ''),
            'date': str(msg['date'] or ''),
            'message_id': str(msg.get('message-id', '')),
        }
        
        # Extract body
        body = self._extract_body(msg)
        parsed['attachments'] = self._extract_attachments(msg)
        
        # Identify vessel from subject/headers
        parsed['vessel'] = self._identify_vessel(parsed, body)
        
        # Split thread into individual emails
        thread_emails_raw = self._split_thread(body)
        
        # REVERSE to get chronological order (oldest first)
        thread_emails_raw.reverse()
        
        # Parse each thread email
        thread_emails = []
        for i, raw_text in enumerate(thread_emails_raw):
            te = ThreadEmail(raw_text, index=i)
            thread_emails.append(te)
        
        parsed['thread_count'] = len(thread_emails)
        parsed['thread_emails'] = [te.to_dict() for te in thread_emails]
        parsed['timeline'] = self._build_timeline(thread_emails, parsed['vessel'])
        
        return parsed
    
    def parse_folder(self, folder_path: str) -> List[Dict]:
        """Parse all EML files in a folder."""
        folder = Path(folder_path)
        results = []
        
        eml_files = sorted(
            list(folder.glob('*.eml')) + list(folder.glob('*.EML')),
            key=lambda f: f.stat().st_mtime
        )
        
        if not eml_files:
            print(f"No .eml files found in {folder_path}")
            return results
        
        print(f"Found {len(eml_files)} EML files")
        
        for eml_file in eml_files:
            try:
                parsed = self.parse_eml(str(eml_file))
                results.append(parsed)
                print(f"  ✓ {eml_file.name} → {parsed.get('vessel', '?')} ({parsed['thread_count']} emails in thread)")
            except Exception as e:
                print(f"  ✗ {eml_file.name} → {e}")
                results.append({'file': str(eml_file.name), 'error': str(e)})
        
        return results
    
    def _extract_body(self, msg) -> str:
        """Extract email body text."""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get('Content-Disposition', ''))
                
                if content_type == 'text/plain' and 'attachment' not in disposition:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body += payload.decode(charset, errors='replace')
                    except Exception:
                        pass
                elif content_type == 'text/html' and not body and 'attachment' not in disposition:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body = self._html_to_text(payload.decode(charset, errors='replace'))
                    except Exception:
                        pass
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    if msg.get_content_type() == 'text/html':
                        body = self._html_to_text(payload.decode(charset, errors='replace'))
                    else:
                        body = payload.decode(charset, errors='replace')
            except Exception:
                body = str(msg.get_payload())
        
        return body
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to text."""
        text = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<div[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _split_thread(self, body: str) -> List[str]:
        # Split on any line that starts with "From:" followed by "Sent:" within a few lines
        # This works across Outlook, Thunderbird, Gmail forwards
        parts = re.split(
            r'(?=^From:\s*.+\r?\n\s*Sent:\s*)',
            body,
            flags=re.MULTILINE
        )
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]
        if not parts:
            parts = [body]
        parts.reverse()  # Chronological order
        return parts
    
    def _extract_attachments(self, msg) -> List[Dict]:
        """Extract attachment metadata."""
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                disposition = str(part.get('Content-Disposition', ''))
                if 'attachment' in disposition:
                    filename = part.get_filename() or 'unknown'
                    size = len(part.get_payload(decode=True) or b'')
                    attachments.append({
                        'filename': filename,
                        'content_type': part.get_content_type(),
                        'size_bytes': size
                    })
        return attachments
    
    def _identify_vessel(self, parsed: Dict, body: str) -> Optional[str]:
        """Identify vessel from email."""
        search_text = f"{parsed['subject']} {parsed['from']} {body[:500]}".lower()
        
        for vessel in KNOWN_VESSELS:
            if vessel.lower() in search_text:
                return vessel.title()
        
        # Try subject pattern
        match = re.search(r'PROJECT[- ](.+?)(?:\s*$|\s*-)', parsed['subject'], re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
        
        # Try "mv" pattern
        match = re.search(r'mv\s+"?([^"]+)"?', body[:1000], re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
        
        return None
    
    def _build_timeline(self, thread_emails: List[ThreadEmail], vessel: str) -> List[Dict]:
        """Build chronological timeline of events."""
        timeline = []
        
        for te in thread_emails:
            if not te.date and not te.stage_updates and not te.blockers:
                continue
            
            entry = {
                'date': te.date.strftime('%Y-%m-%d') if te.date else 'Unknown',
                'sender': te.sender or 'Unknown',
                'sender_role': te.sender_role or 'unknown',
                'products': te.products_mentioned,
                'stage_updates': te.stage_updates,
                'blockers': te.blockers,
                'remarks': te.remarks,
            }
            timeline.append(entry)
        
        return timeline


# =============================================================================
# VESSEL TRACKER v2 - Per-email processing with product association
# =============================================================================

class VesselTrackerV2:
    """Track project stages per vessel with proper product-stage association."""
    
    def __init__(self):
        # vessel -> product -> {stage: {status, date, source, description}}
        self.state = defaultdict(lambda: defaultdict(lambda: OrderedDict([
            ('procurement', {'status': 'pending', 'date': None, 'source': None, 'description': ''}),
            ('install', {'status': 'pending', 'date': None, 'source': None, 'description': ''}),
            ('commission', {'status': 'pending', 'date': None, 'source': None, 'description': ''}),
            ('fat', {'status': 'pending', 'date': None, 'source': None, 'description': ''}),
        ])))
        
        self.history = []
        self.blockers = []
        self.remarks_log = []
    
    def process_thread(self, parsed: Dict):
        """Process all thread emails from a parsed EML file."""
        vessel = parsed.get('vessel')
        if not vessel:
            return
        
        thread_emails = parsed.get('thread_emails', [])
        
        for te_dict in thread_emails:
            self._process_single_email(vessel, te_dict, parsed.get('file', ''))
    
    def _process_single_email(self, vessel: str, te: Dict, source_file: str):
        """Process a single email from the thread."""
        date = te.get('date', '')
        date_short = date[:10] if date else 'Unknown'
        sender = te.get('sender', 'Unknown')
        sender_role = te.get('sender_role', 'unknown')
        products = te.get('products_mentioned', [])
        stages = te.get('stage_updates', {})
        blockers = te.get('blockers', [])
        remarks = te.get('remarks', [])
        
        # If no products detected but stages found, don't apply to all products
        # Instead mark as "General" - this is a key v2 fix
        if not products and stages:
            # Check if we can infer product from context
            # If not, store as general update
            target_products = ['General']
        else:
            target_products = products
        
        # Apply stage updates ONLY to mentioned products
        for product in target_products:
            for stage, info in stages.items():
                new_status = info['status']
                current = self.state[vessel][product][stage]
                
                # MONOTONIC: Only move forward
                if _status_priority(new_status) > _status_priority(current['status']):
                    old_status = current['status']
                    current['status'] = new_status
                    current['date'] = date_short
                    current['source'] = sender
                    current['description'] = info.get('description', '')
                    
                    self.history.append({
                        'date': date_short,
                        'vessel': vessel,
                        'product': product,
                        'stage': stage,
                        'old_status': old_status,
                        'new_status': new_status,
                        'description': info.get('description', ''),
                        'sender': sender,
                        'sender_role': sender_role,
                        'source_file': source_file,
                    })
        
        # Record blockers
        for blocker in blockers:
            self.blockers.append({
                'date': date_short,
                'vessel': vessel,
                'products': products,
                'blocker': blocker,
                'sender': sender,
            })
        
        # Record remarks
        if remarks:
            self.remarks_log.append({
                'date': date_short,
                'vessel': vessel,
                'sender': sender,
                'sender_role': sender_role,
                'remarks': remarks,
            })
    
    def get_summary(self) -> Dict:
        """Get current state."""
        summary = {}
        for vessel, products in self.state.items():
            summary[vessel] = {}
            for product, stages in products.items():
                summary[vessel][product] = {}
                for stage, info in stages.items():
                    summary[vessel][product][stage] = {
                        'status': info['status'],
                        'date': info['date'] or 'N/A',
                        'by': info['source'] or 'N/A',
                    }
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 90)
        print("  VESSEL PROJECT TRACKER v2 - CURRENT STATE")
        print("=" * 90)
        
        for vessel, products in self.state.items():
            print(f"\n{'─' * 90}")
            print(f"  VESSEL: {vessel}")
            print(f"{'─' * 90}")
            
            # Sort products: real products first, General last
            sorted_products = sorted(products.keys(), key=lambda x: (x == 'General', x))
            
            for product in sorted_products:
                stages = products[product]
                print(f"\n  📦 {product}")
                
                for stage, info in stages.items():
                    status = info['status']
                    date = info['date'] or 'N/A'
                    source = info['source'] or ''
                    desc = info['description'] or ''
                    
                    icon = {'pending': '⬚', 'in_progress': '◐', 'done': '✅'}.get(status, '?')
                    status_display = status.upper().replace('_', ' ')
                    
                    line = f"    {icon} {stage.upper():15s} → {status_display:15s} ({date})"
                    if source:
                        line += f"  [{source}]"
                    if desc:
                        line += f"  - {desc}"
                    print(line)
        
        # Print blockers
        if self.blockers:
            print(f"\n{'─' * 90}")
            print("  ⚠️  BLOCKERS / ISSUES:")
            print(f"{'─' * 90}")
            for b in self.blockers:
                products_str = ', '.join(b['products']) if b['products'] else 'General'
                print(f"    [{b['date']}] {products_str}: {b['blocker']} (reported by {b['sender']})")
        
        # Print change history
        if self.history:
            print(f"\n{'─' * 90}")
            print("  📋 CHANGE HISTORY (chronological):")
            print(f"{'─' * 90}")
            for h in self.history:
                role_tag = '🚢' if h['sender_role'] == 'ship' else '🏢'
                print(f"    [{h['date']}] {role_tag} {h['vessel']} | {h['product']} | "
                      f"{h['stage'].upper()} | {h['old_status']} → {h['new_status']} "
                      f"| {h['description']} ({h['sender']})")
        
        print("\n" + "=" * 90)
    
    def export_csv(self, filename: str = None) -> str:
        """Export tracker state to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vessel_tracker_v2_{timestamp}.csv"
        
        rows = []
        for vessel, products in self.state.items():
            for product, stages in products.items():
                row = {'vessel': vessel, 'product': product}
                for stage, info in stages.items():
                    row[f'{stage}_status'] = info['status']
                    row[f'{stage}_date'] = info['date'] or ''
                    row[f'{stage}_by'] = info['source'] or ''
                    row[f'{stage}_notes'] = info['description'] or ''
                rows.append(row)
        
        if rows:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        return filename
    
    def export_history_csv(self, filename: str = None) -> str:
        """Export change history to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"change_history_v2_{timestamp}.csv"
        
        if self.history:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)
        
        return filename
    
    def export_blockers_csv(self, filename: str = None) -> str:
        """Export blockers to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blockers_v2_{timestamp}.csv"
        
        if self.blockers:
            rows = []
            for b in self.blockers:
                rows.append({
                    'date': b['date'],
                    'vessel': b['vessel'],
                    'products': ', '.join(b['products']),
                    'blocker': b['blocker'],
                    'reported_by': b['sender'],
                })
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        return filename


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print('  python eml_parser_v2.py "C:/path/to/email.eml"')
        print('  python eml_parser_v2.py "C:/path/to/folder/"')
        sys.exit(1)
    
    target = sys.argv[1]
    parser = EMLParserV2()
    tracker = VesselTrackerV2()
    
    # Parse
    if os.path.isfile(target):
        print(f"\nParsing: {target}")
        parsed_list = [parser.parse_eml(target)]
    elif os.path.isdir(target):
        print(f"\nParsing folder: {target}")
        parsed_list = parser.parse_folder(target)
    else:
        print(f"Error: '{target}' not found")
        sys.exit(1)
    
    # Process each parsed EML
    for parsed in parsed_list:
        if 'error' in parsed:
            print(f"  ✗ Skipped: {parsed.get('file', '?')} - {parsed['error']}")
            continue
        
        # Feed into tracker
        tracker.process_thread(parsed)
        
        # Print per-email breakdown
        print(f"\n{'═' * 90}")
        print(f"  FILE: {parsed['file']}")
        print(f"  VESSEL: {parsed.get('vessel', 'Unknown')}")
        print(f"  THREAD: {parsed['thread_count']} emails")
        print(f"{'═' * 90}")
        
        for te in parsed.get('thread_emails', []):
            date = te.get('date', 'Unknown')
            if date and date != 'Unknown':
                date = date[:10]
            sender = te.get('sender', '?')
            role = te.get('sender_role', '?')
            products = te.get('products_mentioned', [])
            stages = te.get('stage_updates', {})
            blockers = te.get('blockers', [])
            remarks = te.get('remarks', [])
            
            role_icon = '🚢' if role == 'ship' else '🏢' if role == 'office' else '❓'
            
            print(f"\n  {role_icon} [{date}] {sender}")
            
            if products:
                print(f"     Products: {', '.join(products)}")
            
            if stages:
                for stage, info in stages.items():
                    status_icon = {'done': '✅', 'in_progress': '◐', 'pending': '⬚'}.get(info['status'], '?')
                    print(f"     {status_icon} {stage.upper()}: {info['status']} - {info['description']}")
            
            if blockers:
                for b in blockers:
                    print(f"     ⚠️  BLOCKER: {b}")
            
            if remarks:
                for r in remarks[:2]:
                    print(f"     💬 {r[:100]}")
    
    # Print tracker summary
    tracker.print_summary()
    
    # Export files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON report
    json_file = f"parsed_v2_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_list, f, indent=2, default=str)
    print(f"\n✓ JSON report: {json_file}")
    
    # CSV exports
    csv_file = tracker.export_csv()
    print(f"✓ Tracker CSV: {csv_file}")
    
    history_file = tracker.export_history_csv()
    print(f"✓ History CSV: {history_file}")
    
    if tracker.blockers:
        blockers_file = tracker.export_blockers_csv()
        print(f"✓ Blockers CSV: {blockers_file}")
    
    print(f"\nDone! Processed {len(parsed_list)} EML file(s)")
    print(f"Total thread emails analyzed: {sum(p.get('thread_count', 0) for p in parsed_list if 'error' not in p)}")


if __name__ == "__main__":
    main()