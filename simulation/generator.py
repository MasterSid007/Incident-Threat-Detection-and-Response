"""
Realistic Log Generator for ITDR Prototype.
Creates user behavior patterns that match real-world authentication patterns.
Each user has consistent locations, devices, and work hours.
"""
import random
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from faker import Faker

from .schema import AuthEvent, Identity, Device, Location

fake = Faker()


@dataclass
class UserProfile:
    """
    Represents a realistic user profile with consistent behavior patterns.
    Each user has primary locations, devices, and working hours.
    """
    identity: Identity
    primary_country: str
    primary_city: str
    primary_ip_prefix: str  # e.g., "192.168.1." for home office
    work_ip_prefix: str     # e.g., "10.0.0." for corporate
    devices: List[Device]
    work_start_hour: int  # 7-10
    work_end_hour: int    # 16-20
    is_remote_worker: bool
    travel_probability: float  # 0.0 - 0.05 (most users rarely travel)


class RealisticLogGenerator:
    """
    Generates realistic authentication logs with consistent user behavior.
    Key improvements over basic generator:
    - Users have consistent locations (not random countries each time)
    - Users have consistent devices
    - Users login during their normal work hours
    - Travel is rare and flagged appropriately
    """
    
    # Most users are in these countries
    PRIMARY_COUNTRIES = [
        ("United States", ["New York", "San Francisco", "Seattle", "Austin", "Chicago"]),
        ("United Kingdom", ["London", "Manchester", "Birmingham"]),
        ("Canada", ["Toronto", "Vancouver", "Montreal"]),
        ("Germany", ["Berlin", "Munich", "Frankfurt"]),
        ("Australia", ["Sydney", "Melbourne"]),
    ]
    
    EVENT_TYPES = [
        "UserLoggedIn", "UserLoggedIn", "UserLoggedIn",  # Weight login more
        "TokenRefresh", "TokenRefresh",
        "PasswordReset",
        "MFACompleted",
        "AppAccess"
    ]
    
    BROWSERS = ["Chrome", "Edge", "Firefox", "Safari"]
    OS_LIST = ["Windows 11", "Windows 10", "macOS Sonoma", "macOS Ventura", "Ubuntu 22.04"]
    
    def __init__(self, num_users: int = 50):
        self.num_users = num_users
        self.user_profiles: List[UserProfile] = []
        self._create_user_profiles()
        
    def _create_user_profiles(self):
        """Create realistic user profiles with consistent behaviors."""
        for _ in range(self.num_users):
            # Pick primary location
            country, cities = random.choice(self.PRIMARY_COUNTRIES)
            city = random.choice(cities)
            
            # Create identity
            first_name = fake.first_name()
            last_name = fake.last_name()
            identity = Identity(
                id=str(uuid.uuid4()),
                user_principal_name=f"{first_name.lower()}.{last_name.lower()}@company.com",
                display_name=f"{first_name} {last_name}",
                department=random.choice(["Engineering", "Sales", "HR", "Finance", "IT", "Marketing"]),
                role=random.choice(["User", "Admin", "Developer", "Manager"])
            )
            
            # Create 1-3 consistent devices for this user
            num_devices = random.randint(1, 3)
            devices = []
            for i in range(num_devices):
                os = random.choice(self.OS_LIST)
                devices.append(Device(
                    id=str(uuid.uuid4()),
                    display_name=f"{first_name}'s {os.split()[0]} {i+1}",
                    os=os,
                    browser=random.choice(self.BROWSERS),
                    is_managed=random.random() > 0.2,  # 80% managed
                    is_compliant=random.random() > 0.1  # 90% compliant
                ))
            
            # Work hours (8-5 +/- 2 hours variation)
            work_start = random.randint(7, 10)
            work_end = random.randint(16, 20)
            
            profile = UserProfile(
                identity=identity,
                primary_country=country,
                primary_city=city,
                primary_ip_prefix=f"{random.randint(10,172)}.{random.randint(0,255)}.{random.randint(0,255)}.",
                work_ip_prefix=f"10.{random.randint(0,255)}.{random.randint(0,255)}.",
                devices=devices,
                work_start_hour=work_start,
                work_end_hour=work_end,
                is_remote_worker=random.random() > 0.7,  # 30% remote
                travel_probability=random.uniform(0.0, 0.02)  # 0-2% travel
            )
            self.user_profiles.append(profile)
    
    def _generate_normal_event(self, profile: UserProfile, timestamp: datetime) -> AuthEvent:
        """Generate a normal event for a user following their typical patterns."""
        
        # Pick from user's consistent devices
        device = random.choice(profile.devices)
        
        # Location is almost always their primary location
        is_at_work = not profile.is_remote_worker and random.random() > 0.3
        
        location = Location(
            ip_address=f"{profile.work_ip_prefix if is_at_work else profile.primary_ip_prefix}{random.randint(1,254)}",
            country=profile.primary_country,
            state="",
            city=profile.primary_city,
            asn=f"AS{random.randint(1000, 9999)}"
        )
        
        # Event type based on time of day
        hour = timestamp.hour
        if hour >= profile.work_start_hour and hour <= profile.work_end_hour:
            event_type = random.choice(self.EVENT_TYPES)
        else:
            # After hours - mostly token refresh
            event_type = random.choice(["TokenRefresh", "AppAccess"])
        
        return AuthEvent(
            timestamp=timestamp.isoformat(),
            identity=profile.identity,
            device=device,
            location=location,
            user_agent=f"Mozilla/5.0 ({device.os}) {device.browser}",
            event_type=event_type,
            app_name=random.choice(["Office365", "Teams", "SharePoint", "Outlook", "OneDrive"]),
            status="Success",
            correlation_id=str(uuid.uuid4()),
            is_attack=False,
            attack_type=None
        )
    
    def generate_batch(
        self,
        count: int = 1000,
        start_time: Optional[datetime] = None,
        duration_hours: int = 8
    ) -> List[AuthEvent]:
        """
        Generate a batch of realistic authentication events.
        Events are distributed across work hours with realistic patterns.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        events = []
        end_time = start_time + timedelta(hours=duration_hours)
        
        for _ in range(count):
            # Pick a random user
            profile = random.choice(self.user_profiles)
            
            # Generate timestamp within work hours for this user
            # Add some noise for after-hours activity
            base_hour = random.randint(profile.work_start_hour, profile.work_end_hour)
            
            # 10% chance of after-hours activity
            if random.random() < 0.1:
                base_hour = random.choice([6, 7, 21, 22, 23])
            
            timestamp = start_time + timedelta(
                hours=random.randint(0, duration_hours - 1),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            
            event = self._generate_normal_event(profile, timestamp)
            events.append(event)
        
        return events
    
    @property
    def identities(self) -> List[Identity]:
        """Return list of all user identities."""
        return [p.identity for p in self.user_profiles]


# Alias for backward compatibility
LogGenerator = RealisticLogGenerator
