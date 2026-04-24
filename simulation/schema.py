"""
Data models for ITDR log events.
Defines the schema for simulated Entra ID/Okta-style authentication telemetry.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import json
import uuid


@dataclass
class Identity:
    """Represents a user or service principal identity."""
    id: str  # GUID
    user_principal_name: str  # email/UPN
    display_name: str
    role: str  # 'User', 'Admin', 'ServicePrincipal'
    department: str


@dataclass
class Device:
    """Represents a device used for authentication."""
    id: str
    display_name: str
    os: str
    browser: str
    is_managed: bool
    is_compliant: bool


@dataclass
class Location:
    """Represents the geographic location of an authentication event."""
    ip_address: str
    country: str
    state: str
    city: str
    asn: str  # Autonomous System Number


@dataclass
class AuthEvent:
    """
    Represents a single authentication or identity event.
    
    This is the core data model that flows through the detection pipeline.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Actor
    identity: Optional[Identity] = None
    
    # Context
    device: Optional[Device] = None
    location: Optional[Location] = None
    user_agent: str = ""
    
    # Event Details
    event_type: str = "UserLoggedIn"  # UserLoggedIn, UserLoginFailed, TokenRefresh, RoleAssigned, AdminAction, BulkDownload
    app_name: str = "Office 365"
    status: str = "Success"  # Success, Failure
    failure_reason: Optional[str] = None
    
    # Risk Labels (Ground Truth for training/evaluation)
    is_attack: bool = False
    attack_type: Optional[str] = None  # PasswordSpray, ImpossibleTravel, TokenTheft, PrivilegeEscalation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "correlationId": self.correlation_id,
            "identity": {
                "id": self.identity.id if self.identity else None,
                "upn": self.identity.user_principal_name if self.identity else None,
                "displayName": self.identity.display_name if self.identity else None,
                "role": self.identity.role if self.identity else None,
                "department": self.identity.department if self.identity else None
            } if self.identity else {},
            "device": {
                "id": self.device.id if self.device else None,
                "displayName": self.device.display_name if self.device else None,
                "os": self.device.os if self.device else None,
                "browser": self.device.browser if self.device else None,
                "isManaged": self.device.is_managed if self.device else None,
                "isCompliant": self.device.is_compliant if self.device else None
            } if self.device else {},
            "location": {
                "ip": self.location.ip_address if self.location else None,
                "country": self.location.country if self.location else None,
                "state": self.location.state if self.location else None,
                "city": self.location.city if self.location else None,
                "asn": self.location.asn if self.location else None
            } if self.location else {},
            "userAgent": self.user_agent,
            "eventType": self.event_type,
            "appName": self.app_name,
            "status": self.status,
            "failureReason": self.failure_reason,
            "label": {
                "isAttack": self.is_attack,
                "attackType": self.attack_type
            }
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())
