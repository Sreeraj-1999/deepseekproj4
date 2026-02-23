"""
EML Mail Parser for Project Tracking
=====================================
Run this on your machine against your .eml files.
It extracts structure, threads, vessel references, and stage indicators.

Usage:
    python eml_parser.py <path_to_eml_file_or_folder>
    
    # Single file
    python eml_parser.py "C:/Users/User/Desktop/outlook.eml"
    
    # Folder of EML files
    python eml_parser.py "C:/Users/User/Desktop/eml_exports/"

Output:
    - Prints parsed summary to console
    - Saves JSON report to parsed_emails_<timestamp>.json
    - Saves CSV tracker to vessel_tracker_<timestamp>.csv
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
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


# =============================================================================
# CONFIGURATION - Edit these for your setup
# =============================================================================

# Known vessel names (add yours here)
KNOWN_VESSELS = [
    "molly schulte",
    "clemens schulte", 
    "maren schulte",
    "anna schulte",
    # Add more vessels as needed
]

# Known products
PRODUCTS = ["MDC", "Transbox", "Electrical Panel", "Flowmeter", "Energy Meter", "SPM"]

# Stage keywords - maps natural language to stages
STAGE_INDICATORS = {
    "procurement": {
        "done": [
            "parts arrived", "received the", "delivered", "parts on board",
            "equipment received", "hardware received", "shipped to vessel",
            "goods received", "material on board", "procurement completed",
            "items received", "cargo received"
        ],
        "in_progress": [
            "ordered", "waiting for delivery", "shipment", "dispatch",
            "procurement in progress", "purchase order", "PO raised",
            "awaiting delivery", "in transit", "being shipped"
        ],
        "pending": [
            "yet to order", "not ordered", "procurement pending",
            "awaiting approval", "budget approval"
        ]
    },
    "install": {
        "done": [
            "installed", "installation completed", "connections were made",
            "connected to port", "mounted", "fitted", "installation done",
            "hardware installed", "physically installed", "cabling done",
            "wiring completed", "panel installed"
        ],
        "in_progress": [
            "installing", "installation in progress", "being installed",
            "partially installed", "installation ongoing", "fitting in progress"
        ],
        "pending": [
            "yet to install", "installation pending", "not installed",
            "awaiting installation", "install pending"
        ]
    },
    "commission": {
        "done": [
            "commissioned", "commissioning completed", "system online",
            "remote access working", "data flowing", "telemetry active",
            "system operational", "commission done", "successfully configured",
            "connection established"
        ],
        "in_progress": [
            "unable to access", "trying to access", "troubleshooting",
            "network issue", "configuring", "commissioning in progress",
            "testing connection", "powered on", "not powered on",
            "login", "credentials", "ip address", "screenshot",
            "checking connectivity", "remote access", "trying to connect"
        ],
        "pending": [
            "yet to commission", "commissioning pending", "not commissioned",
            "awaiting commissioning"
        ]
    },
    "fat": {
        "done": [
            "FAT completed", "factory acceptance", "FAT done",
            "acceptance test passed", "FAT signed off", "test completed"
        ],
        "in_progress": [
            "FAT in progress", "FAT scheduled", "testing in progress",
            "acceptance testing", "FAT ongoing"
        ],
        "pending": [
            "FAT pending", "yet to conduct FAT", "FAT not done",
            "awaiting FAT"
        ]
    }
}

# Signature/noise patterns to strip
NOISE_PATTERNS = [
    r"Thanks\s*[&and]*\s*Regards.*",
    r"Sincerely,.*",
    r"Best\s+regards.*",
    r"Kind\s+regards.*",
    r"MariApps\s+Marine\s+Solutions.*",
    r"MariApps\s+House.*",
    r"www\.mariapps\.com.*",
    r"Tel(?:ephone)?:.*",
    r"Telex:.*",
    r"FBB\s+Phone:.*",
    r"IP\s+Phone.*",
    r"Email:\s*__.*",
    r"\|.*(?:Engineer|Manager|Director|Officer).*@.*",
    r"Plot\s+No\s+A2.*",
    r"SmartCity\s+Kochi.*",
    r"Kerala\s*–?\s*\d+",
]


# =============================================================================
# EML PARSER
# =============================================================================

class EMLParser:
    """Parse .eml files and extract structured data."""
    
    def __init__(self):
        self.noise_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NOISE_PATTERNS]
    
    def parse_eml(self, file_path: str) -> Dict:
        """Parse a single .eml file."""
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        # Extract headers
        parsed = {
            'file': str(file_path.name),
            'subject': str(msg['subject'] or ''),
            'from': str(msg['from'] or ''),
            'to': str(msg['to'] or ''),
            'cc': str(msg['cc'] or ''),
            'date': str(msg['date'] or ''),
            'message_id': str(msg['message-id'] or ''),
            'in_reply_to': str(msg.get('in-reply-to', '')),
            'references': str(msg.get('references', '')),
        }
        
        # Parse date
        parsed['parsed_date'] = self._parse_date(parsed['date'])
        
        # Extract body
        body = self._extract_body(msg)
        parsed['raw_body'] = body
        parsed['clean_body'] = self._clean_body(body)
        
        # Extract individual emails from thread
        parsed['thread_emails'] = self._split_thread(body)
        
        # Extract attachments info
        parsed['attachments'] = self._extract_attachments(msg)
        
        # Identify vessel
        parsed['vessel'] = self._identify_vessel(parsed)
        
        # Identify products mentioned
        parsed['products_mentioned'] = self._identify_products(body)
        
        # Identify stage indicators
        parsed['stage_indicators'] = self._identify_stages(body)
        
        # Extract people
        parsed['people'] = self._extract_people(parsed)
        
        # Generate remarks
        parsed['remarks'] = self._generate_remarks(parsed)
        
        return parsed
    
    def parse_folder(self, folder_path: str) -> List[Dict]:
        """Parse all .eml files in a folder."""
        folder = Path(folder_path)
        results = []
        
        eml_files = list(folder.glob('*.eml')) + list(folder.glob('*.EML'))
        
        if not eml_files:
            print(f"No .eml files found in {folder_path}")
            return results
        
        print(f"Found {len(eml_files)} EML files")
        
        for eml_file in sorted(eml_files):
            try:
                parsed = self.parse_eml(str(eml_file))
                results.append(parsed)
                print(f"  ✓ Parsed: {eml_file.name} → Vessel: {parsed.get('vessel', 'Unknown')}")
            except Exception as e:
                print(f"  ✗ Failed: {eml_file.name} → {e}")
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
                            html = payload.decode(charset, errors='replace')
                            body = self._html_to_text(html)
                    except Exception:
                        pass
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    content_type = msg.get_content_type()
                    if content_type == 'text/html':
                        body = self._html_to_text(payload.decode(charset, errors='replace'))
                    else:
                        body = payload.decode(charset, errors='replace')
            except Exception:
                body = str(msg.get_payload())
        
        return body
    
    def _html_to_text(self, html: str) -> str:
        """Simple HTML to text conversion."""
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
    
    def _clean_body(self, body: str) -> str:
        """Remove signatures, disclaimers, noise."""
        clean = body
        for pattern in self.noise_patterns:
            clean = pattern.sub('', clean)
        
        # Remove excessive whitespace
        clean = re.sub(r'\n{3,}', '\n\n', clean)
        clean = re.sub(r'[ \t]+', ' ', clean)
        clean = re.sub(r'_{3,}', '', clean)
        
        return clean.strip()
    
    def _split_thread(self, body: str) -> List[Dict]:
        """Split email thread into individual messages."""
        # Common reply separators
        separators = [
            r'From:.*?Sent:.*?(?:To:|Subject:)',
            r'-{3,}\s*Original\s+Message\s*-{3,}',
            r'On\s+\w+,\s+\w+\s+\d+,\s+\d+.*?wrote:',
        ]
        
        combined_pattern = '|'.join(f'({sep})' for sep in separators)
        
        parts = re.split(combined_pattern, body, flags=re.IGNORECASE | re.DOTALL)
        
        thread_emails = []
        current_text = ""
        
        for part in parts:
            if part is None:
                continue
            
            # Check if this is a separator
            is_separator = False
            for sep in separators:
                if re.match(sep, part.strip(), re.IGNORECASE | re.DOTALL):
                    is_separator = True
                    break
            
            if is_separator:
                if current_text.strip():
                    thread_emails.append({
                        'text': current_text.strip(),
                        'length': len(current_text.strip())
                    })
                current_text = part
            else:
                current_text += part
        
        if current_text.strip():
            thread_emails.append({
                'text': current_text.strip(),
                'length': len(current_text.strip())
            })
        
        # If no splits found, return whole body as single email
        if not thread_emails:
            thread_emails = [{'text': body.strip(), 'length': len(body.strip())}]
        
        return thread_emails
    
    def _extract_attachments(self, msg) -> List[Dict]:
        """Extract attachment metadata (not content)."""
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
    
    def _identify_vessel(self, parsed: Dict) -> Optional[str]:
        """Identify which vessel this email is about."""
        # Check subject first (most reliable)
        search_text = f"{parsed['subject']} {parsed['from']} {parsed['to']} {parsed['raw_body'][:500]}"
        search_lower = search_text.lower()
        
        for vessel in KNOWN_VESSELS:
            if vessel.lower() in search_lower:
                return vessel.title()
        
        # Try to extract vessel name from subject pattern like "PROJECT-VESSEL NAME"
        subject = parsed['subject']
        match = re.search(r'PROJECT[- ](.+?)(?:\s*$|\s*-)', subject, re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
        
        # Try "mv " pattern (motor vessel)
        match = re.search(r'mv\s+"?([^"]+)"?', search_text, re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
        
        return None
    
    def _identify_products(self, body: str) -> List[str]:
        """Identify which products are mentioned."""
        body_lower = body.lower()
        found = []
        
        product_patterns = {
            "MDC": [r'\bMDC\b', r'mdc\s+pc', r'data\s+collector'],
            "Transbox": [r'\btransbox\b', r'\btrans\s*box\b'],
            "Electrical Panel": [r'electrical\s+panel', r'\bpanel\b'],
            "Flowmeter": [r'flow\s*meter', r'\bflowmeter\b'],
            "Energy Meter": [r'energy\s*meter', r'\benergy\s+meter\b'],
            "SPM": [r'\bSPM\b', r'shaft\s+power'],
        }
        
        for product, patterns in product_patterns.items():
            for pattern in patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    found.append(product)
                    break
        
        return found
    
    def _identify_stages(self, body: str) -> Dict:
        """Identify stage indicators from email content."""
        body_lower = body.lower()
        found_stages = {}
        
        for stage, status_keywords in STAGE_INDICATORS.items():
            for status, keywords in status_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in body_lower:
                        if stage not in found_stages:
                            found_stages[stage] = {
                                'status': status,
                                'matched_keyword': keyword
                            }
                        # Don't override "done" with lesser status
                        elif self._status_priority(status) > self._status_priority(found_stages[stage]['status']):
                            found_stages[stage] = {
                                'status': status,
                                'matched_keyword': keyword
                            }
        
        return found_stages
    
    def _status_priority(self, status: str) -> int:
        """Higher number = more advanced status."""
        return {"pending": 0, "in_progress": 1, "done": 2}.get(status, -1)
    
    def _extract_people(self, parsed: Dict) -> List[Dict]:
        """Extract people mentioned in the email."""
        people = []
        
        # From header
        from_match = re.search(r'([^<]+)<([^>]+)>', parsed['from'])
        if from_match:
            people.append({
                'name': from_match.group(1).strip().strip('"'),
                'email': from_match.group(2).strip(),
                'role': 'sender'
            })
        
        # Look for Captain/CE mentions
        body = parsed['raw_body']
        
        captain_match = re.search(r'(?:Capt\.?|Captain)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', body)
        if captain_match:
            people.append({
                'name': captain_match.group(1),
                'role': 'captain'
            })
        
        ce_match = re.search(r'CE\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', body)
        if ce_match:
            people.append({
                'name': ce_match.group(1),
                'role': 'chief_engineer'
            })
        
        return people
    
    def _generate_remarks(self, parsed: Dict) -> List[str]:
        """Generate human-readable remarks from the email."""
        remarks = []
        clean = parsed['clean_body']
        
        # Take first meaningful sentences (skip greetings)
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        
        for line in lines:
            # Skip greetings and noise
            if re.match(r'^(Dear|Good\s+day|Hi|Hello|Noted)', line, re.IGNORECASE):
                continue
            if len(line) < 10:
                continue
            if re.match(r'^(From:|To:|Cc:|Subject:|Sent:)', line, re.IGNORECASE):
                continue
            
            # This is likely meaningful content
            if len(line) < 200:
                remarks.append(line)
            
            if len(remarks) >= 3:
                break
        
        return remarks
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse email date to ISO format."""
        if not date_str:
            return None
        
        # Common email date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S',
            '%d %B %Y %H:%M',
        ]
        
        # Clean the date string
        clean_date = re.sub(r'\(.*?\)', '', date_str).strip()
        
        for fmt in formats:
            try:
                dt = datetime.strptime(clean_date, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        return date_str  # Return original if parsing fails


# =============================================================================
# VESSEL TRACKER - Accumulates state across emails
# =============================================================================

class VesselTracker:
    """Track project stages per vessel across multiple emails."""
    
    def __init__(self):
        # vessel -> product -> {stage: {status, date, source}}
        self.state = defaultdict(lambda: defaultdict(lambda: {
            'procurement': {'status': 'pending', 'date': None, 'source': None},
            'install': {'status': 'pending', 'date': None, 'source': None},
            'commission': {'status': 'pending', 'date': None, 'source': None},
            'fat': {'status': 'pending', 'date': None, 'source': None},
        }))
        self.history = []  # Full audit trail
    
    def update_from_parsed_email(self, parsed: Dict):
        """Update tracker state from a parsed email."""
        vessel = parsed.get('vessel')
        if not vessel:
            return
        
        date = parsed.get('parsed_date', parsed.get('date', 'unknown'))
        products = parsed.get('products_mentioned', [])
        stages = parsed.get('stage_indicators', {})
        
        if not products:
            products = ['General']  # Track even if no specific product mentioned
        
        for product in products:
            for stage, info in stages.items():
                new_status = info['status']
                current = self.state[vessel][product][stage]
                
                # MONOTONIC: Only move forward (pending -> in_progress -> done)
                priority = {'pending': 0, 'in_progress': 1, 'done': 2}
                
                if priority.get(new_status, -1) > priority.get(current['status'], -1):
                    old_status = current['status']
                    current['status'] = new_status
                    current['date'] = date
                    current['source'] = parsed.get('file', 'unknown')
                    
                    self.history.append({
                        'timestamp': date,
                        'vessel': vessel,
                        'product': product,
                        'stage': stage,
                        'old_status': old_status,
                        'new_status': new_status,
                        'matched_keyword': info.get('matched_keyword', ''),
                        'source_file': parsed.get('file', 'unknown'),
                        'remarks': '; '.join(parsed.get('remarks', []))
                    })
    
    def get_summary(self) -> Dict:
        """Get current state of all vessels."""
        summary = {}
        for vessel, products in self.state.items():
            summary[vessel] = {}
            for product, stages in products.items():
                summary[vessel][product] = {
                    stage: f"{info['status']} ({info['date'][:10] if info['date'] and info['date'] != 'unknown' else 'N/A'})"
                    for stage, info in stages.items()
                }
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 80)
        print("VESSEL PROJECT TRACKER - CURRENT STATE")
        print("=" * 80)
        
        for vessel, products in self.state.items():
            print(f"\n{'─' * 80}")
            print(f"  VESSEL: {vessel}")
            print(f"{'─' * 80}")
            
            for product, stages in products.items():
                print(f"\n  Product: {product}")
                for stage, info in stages.items():
                    status = info['status'].upper()
                    date = info['date'][:10] if info['date'] and info['date'] != 'unknown' else 'N/A'
                    
                    # Status indicator
                    icon = {'pending': '○', 'in_progress': '◐', 'done': '●'}.get(info['status'], '?')
                    print(f"    {icon} {stage.upper():15s} → {status:15s} ({date})")
        
        if self.history:
            print(f"\n{'─' * 80}")
            print("  CHANGE HISTORY:")
            print(f"{'─' * 80}")
            for entry in self.history:
                print(f"    [{entry['timestamp'][:10] if entry['timestamp'] else 'N/A'}] "
                      f"{entry['vessel']} | {entry['product']} | {entry['stage']} | "
                      f"{entry['old_status']} → {entry['new_status']} "
                      f"(matched: '{entry['matched_keyword']}')")
        
        print("\n" + "=" * 80)
    
    def export_csv(self, filename: str = None) -> str:
        """Export tracker to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vessel_tracker_{timestamp}.csv"
        
        rows = []
        for vessel, products in self.state.items():
            for product, stages in products.items():
                row = {
                    'vessel': vessel,
                    'product': product,
                }
                for stage, info in stages.items():
                    row[f'{stage}_status'] = info['status']
                    row[f'{stage}_date'] = info['date'][:10] if info['date'] and info['date'] != 'unknown' else ''
                rows.append(row)
        
        if rows:
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
        print("\nNo path provided. Example usage:")
        print('  python eml_parser.py "C:/Users/User/Desktop/emails/"')
        print('  python eml_parser.py "C:/Users/User/Desktop/email.eml"')
        sys.exit(1)
    
    target = sys.argv[1]
    parser = EMLParser()
    tracker = VesselTracker()
    
    # Parse
    if os.path.isfile(target):
        print(f"\nParsing single file: {target}")
        parsed_emails = [parser.parse_eml(target)]
    elif os.path.isdir(target):
        print(f"\nParsing folder: {target}")
        parsed_emails = parser.parse_folder(target)
    else:
        print(f"Error: '{target}' not found")
        sys.exit(1)
    
    # Process each email
    for parsed in parsed_emails:
        if 'error' in parsed:
            continue
        
        tracker.update_from_parsed_email(parsed)
        
        # Print individual email summary
        print(f"\n{'─' * 60}")
        print(f"File:     {parsed['file']}")
        print(f"Subject:  {parsed['subject'][:80]}")
        print(f"From:     {parsed['from'][:60]}")
        print(f"Date:     {parsed.get('parsed_date', 'N/A')}")
        print(f"Vessel:   {parsed.get('vessel', 'Unknown')}")
        print(f"Products: {', '.join(parsed.get('products_mentioned', [])) or 'None detected'}")
        print(f"Stages:   {json.dumps(parsed.get('stage_indicators', {}), indent=2)}")
        print(f"Remarks:  {parsed.get('remarks', [])}")
        print(f"Thread emails: {len(parsed.get('thread_emails', []))}")
        print(f"Attachments: {len(parsed.get('attachments', []))}")
        
        for att in parsed.get('attachments', []):
            print(f"  - {att['filename']} ({att['content_type']}, {att['size_bytes']} bytes)")
    
    # Print tracker summary
    tracker.print_summary()
    
    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    json_file = f"parsed_emails_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        # Remove raw_body for cleaner output
        export_data = []
        for p in parsed_emails:
            clean = {k: v for k, v in p.items() if k != 'raw_body'}
            export_data.append(clean)
        json.dump(export_data, f, indent=2, default=str)
    print(f"\n✓ JSON report saved: {json_file}")
    
    # Save CSV
    csv_file = tracker.export_csv()
    print(f"✓ CSV tracker saved: {csv_file}")
    
    # Save history
    if tracker.history:
        history_file = f"change_history_{timestamp}.csv"
        with open(history_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=tracker.history[0].keys())
            writer.writeheader()
            writer.writerows(tracker.history)
        print(f"✓ Change history saved: {history_file}")
    
    print(f"\nDone! Processed {len(parsed_emails)} emails.")


if __name__ == "__main__":
    main()