"""
Attack Scenario Simulator.
Generates realistic attack patterns for testing detection capabilities.
"""
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .schema import AuthEvent, Identity, Device, Location
from .generator import RealisticLogGenerator, UserProfile


class AttackSimulator:
    """
    Simulates various identity-based attack patterns.
    Uses the realistic generator to create authentic-looking attack traffic.
    """
    
    # High-risk countries for attack origination
    HIGH_RISK_COUNTRIES = ["Russia", "China", "North Korea", "Iran"]
    HIGH_RISK_CITIES = {
        "Russia": ["Moscow", "St. Petersburg"],
        "China": ["Beijing", "Shanghai"],
        "North Korea": ["Pyongyang"],
        "Iran": ["Tehran"]
    }
    
    ATTACK_USER_AGENTS = [
        "python-requests/2.28.0",
        "curl/7.68.0",
        "Mozilla/5.0 (compatible; Headless Chrome/100.0)",
        "Go-http-client/1.1"
    ]
    
    def __init__(self, generator: RealisticLogGenerator):
        self.gen = generator
        
    def _get_random_profile(self) -> UserProfile:
        """Get a random user profile from the generator."""
        return random.choice(self.gen.user_profiles)
    
    def _create_attack_location(self) -> Location:
        """Create a location from a high-risk country."""
        country = random.choice(self.HIGH_RISK_COUNTRIES)
        city = random.choice(self.HIGH_RISK_CITIES[country])
        return Location(
            ip_address=f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            country=country,
            state="",
            city=city,
            asn=f"AS{random.randint(10000, 99999)}"
        )
    
    def _create_attacker_device(self) -> Device:
        """Create a suspicious unmanaged device."""
        return Device(
            id=str(uuid.uuid4()),
            display_name="Unknown Device",
            os=random.choice(["Linux", "Kali Linux", "Unknown"]),
            browser=random.choice(["curl", "python-requests", "Headless Chrome"]),
            is_managed=False,
            is_compliant=False
        )
    
    def simulate_password_spray(
        self,
        target_count: int = 10,
        start_time: Optional[datetime] = None
    ) -> List[AuthEvent]:
        """
        Simulate a password spray attack.
        Multiple users targeted from same IP with failed logins.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        events = []
        attacker_ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        attack_location = self._create_attack_location()
        attack_location.ip_address = attacker_ip
        attacker_device = self._create_attacker_device()
        
        # Target random subset of users
        targets = random.sample(self.gen.user_profiles, min(target_count, len(self.gen.user_profiles)))
        current_time = start_time
        
        for profile in targets:
            # 1-3 failed attempts per user
            attempts = random.randint(1, 3)
            for _ in range(attempts):
                event = AuthEvent(
                    timestamp=current_time.isoformat(),
                    identity=profile.identity,
                    device=attacker_device,
                    location=attack_location,
                    user_agent=random.choice(self.ATTACK_USER_AGENTS),
                    event_type="UserLoginFailed",
                    app_name="Office365",
                    status="Failure",
                    correlation_id=str(uuid.uuid4()),
                    is_attack=True,
                    attack_type="PasswordSpray"
                )
                events.append(event)
                current_time += timedelta(seconds=random.randint(2, 10))
        
        return events
    
    def simulate_impossible_travel(
        self,
        start_time: Optional[datetime] = None
    ) -> List[AuthEvent]:
        """
        Simulate an impossible travel attack.
        User logs in from legitimate location, then from foreign country within minutes.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        profile = self._get_random_profile()
        events = []
        
        # First: legitimate login from user's normal location
        normal_device = random.choice(profile.devices)
        normal_location = Location(
            ip_address=f"{profile.primary_ip_prefix}{random.randint(1,254)}",
            country=profile.primary_country,
            state="",
            city=profile.primary_city,
            asn=f"AS{random.randint(1000, 9999)}"
        )
        
        event1 = AuthEvent(
            timestamp=start_time.isoformat(),
            identity=profile.identity,
            device=normal_device,
            location=normal_location,
            user_agent=f"Mozilla/5.0 ({normal_device.os}) {normal_device.browser}",
            event_type="UserLoggedIn",
            app_name="Office365",
            status="Success",
            correlation_id=str(uuid.uuid4()),
            is_attack=False,
            attack_type=None
        )
        events.append(event1)
        
        # Second: suspicious login from high-risk country within 15 minutes
        attack_time = start_time + timedelta(minutes=random.randint(5, 15))
        attack_location = self._create_attack_location()
        attack_device = self._create_attacker_device()
        
        event2 = AuthEvent(
            timestamp=attack_time.isoformat(),
            identity=profile.identity,
            device=attack_device,
            location=attack_location,
            user_agent=random.choice(self.ATTACK_USER_AGENTS),
            event_type="UserLoggedIn",
            app_name="Office365",
            status="Success",
            correlation_id=str(uuid.uuid4()),
            is_attack=True,
            attack_type="ImpossibleTravel"
        )
        events.append(event2)
        
        return events
    
    def simulate_token_theft(
        self,
        start_time: Optional[datetime] = None
    ) -> List[AuthEvent]:
        """
        Simulate token theft - same session used from different IP/device.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        profile = self._get_random_profile()
        events = []
        session_id = str(uuid.uuid4())  # Same session
        
        # First: legitimate token use
        normal_device = random.choice(profile.devices)
        normal_location = Location(
            ip_address=f"{profile.primary_ip_prefix}{random.randint(1,254)}",
            country=profile.primary_country,
            state="",
            city=profile.primary_city,
            asn=f"AS{random.randint(1000, 9999)}"
        )
        
        event1 = AuthEvent(
            timestamp=start_time.isoformat(),
            identity=profile.identity,
            device=normal_device,
            location=normal_location,
            user_agent=f"Mozilla/5.0 ({normal_device.os}) {normal_device.browser}",
            event_type="TokenRefresh",
            app_name="Office365",
            status="Success",
            correlation_id=session_id,
            is_attack=False,
            attack_type=None
        )
        events.append(event1)
        
        # Second: same session from different location (token stolen)
        attack_time = start_time + timedelta(minutes=random.randint(1, 5))
        attack_location = self._create_attack_location()
        attack_device = self._create_attacker_device()
        
        event2 = AuthEvent(
            timestamp=attack_time.isoformat(),
            identity=profile.identity,
            device=attack_device,
            location=attack_location,
            user_agent=random.choice(self.ATTACK_USER_AGENTS),
            event_type="TokenRefresh",
            app_name="Office365",
            status="Success",
            correlation_id=session_id,  # Same session!
            is_attack=True,
            attack_type="TokenTheft"
        )
        events.append(event2)
        
        return events
    
    def simulate_privilege_escalation(
        self,
        start_time: Optional[datetime] = None
    ) -> List[AuthEvent]:
        """
        Simulate privilege escalation - role assignment followed by suspicious activity.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        profile = self._get_random_profile()
        events = []
        
        # Role assignment event
        event1 = AuthEvent(
            timestamp=start_time.isoformat(),
            identity=profile.identity,
            device=random.choice(profile.devices),
            location=Location(
                ip_address=f"{profile.primary_ip_prefix}{random.randint(1,254)}",
                country=profile.primary_country,
                state="",
                city=profile.primary_city,
                asn=f"AS{random.randint(1000, 9999)}"
            ),
            user_agent=f"Mozilla/5.0 Windows Chrome",
            event_type="RoleAssigned",
            app_name="AzureAD",
            status="Success",
            correlation_id=str(uuid.uuid4()),
            is_attack=True,
            attack_type="PrivilegeEscalation"
        )
        events.append(event1)
        
        # Suspicious admin activity shortly after
        attack_time = start_time + timedelta(minutes=random.randint(2, 10))
        attack_location = self._create_attack_location()
        
        event2 = AuthEvent(
            timestamp=attack_time.isoformat(),
            identity=profile.identity,
            device=self._create_attacker_device(),
            location=attack_location,
            user_agent=random.choice(self.ATTACK_USER_AGENTS),
            event_type="AdminActivity",
            app_name="AzureAD",
            status="Success",
            correlation_id=str(uuid.uuid4()),
            is_attack=True,
            attack_type="PrivilegeEscalation"
        )
        events.append(event2)
        
        # Additional suspicious activity
        event3 = AuthEvent(
            timestamp=(attack_time + timedelta(minutes=1)).isoformat(),
            identity=profile.identity,
            device=self._create_attacker_device(),
            location=attack_location,
            user_agent=random.choice(self.ATTACK_USER_AGENTS),
            event_type="BulkDownload",
            app_name="SharePoint",
            status="Success",
            correlation_id=str(uuid.uuid4()),
            is_attack=True,
            attack_type="PrivilegeEscalation"
        )
        events.append(event3)
        
        return events
