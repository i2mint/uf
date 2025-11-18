"""Full-featured example demonstrating all uf capabilities.

This example showcases:
- UI decorators (@ui_config, @group, etc.)
- Field configurations (email, textarea, etc.)
- Function grouping and organization
- Custom type transformations
- Field dependencies
- Testing utilities

To run this example:
    python examples/full_featured_example.py
"""

from datetime import datetime, date
from typing import Optional
from uf import (
    # Core
    mk_grouped_app,
    # Decorators
    ui_config,
    group,
    field_config,
    with_example,
    # Configuration
    RjsfFieldConfig,
    get_field_config,
    # Organization
    FunctionGroup,
    # Transformation
    InputTransformRegistry,
    # Field interactions
    FieldDependency,
    DependencyAction,
    with_dependencies,
)


# =============================================================================
# User Management Functions (Admin Group)
# =============================================================================

@ui_config(
    title="Create New User",
    description="Register a new user in the system",
    group="Admin",
    icon="user-plus",
    order=1,
)
@field_config(
    email=get_field_config('email'),
    bio=get_field_config('multiline_text'),
)
@with_example(
    name="John Doe",
    email="john@example.com",
    age=30,
    bio="Software developer with 10 years of experience"
)
def create_user(
    name: str,
    email: str,
    age: int,
    bio: str = "",
    is_admin: bool = False,
) -> dict:
    """Create a new user account.

    Args:
        name: Full name of the user
        email: Email address
        age: Age in years
        bio: Short biography
        is_admin: Whether user has admin privileges

    Returns:
        Dictionary with user details and creation timestamp
    """
    return {
        "id": hash(email) % 10000,  # Fake ID for demo
        "name": name,
        "email": email,
        "age": age,
        "bio": bio,
        "is_admin": is_admin,
        "created_at": datetime.now().isoformat(),
    }


@group("Admin")
@with_example(user_id=1234)
def delete_user(user_id: int) -> dict:
    """Delete a user from the system.

    Args:
        user_id: ID of the user to delete

    Returns:
        Confirmation message
    """
    return {
        "success": True,
        "message": f"User {user_id} has been deleted",
        "deleted_at": datetime.now().isoformat(),
    }


# =============================================================================
# Reporting Functions (Reports Group)
# =============================================================================

@ui_config(
    title="Generate Report",
    group="Reports",
    icon="file-text",
    order=1,
)
@field_config(
    report_type=RjsfFieldConfig(
        widget='select',
        enum=['daily', 'weekly', 'monthly', 'yearly'],
    ),
    format=RjsfFieldConfig(
        widget='radio',
        enum=['pdf', 'csv', 'excel'],
    ),
    start_date=get_field_config('date'),
    end_date=get_field_config('date'),
)
def generate_report(
    report_type: str,
    format: str = 'pdf',
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    """Generate a report for the specified period.

    Args:
        report_type: Type of report to generate
        format: Output format
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        Report metadata and download link
    """
    return {
        "report_type": report_type,
        "format": format,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "generated_at": datetime.now().isoformat(),
        "download_url": f"/downloads/report_{datetime.now().timestamp()}.{format}",
    }


@group("Reports")
def export_data(
    data_type: str,
    include_archived: bool = False,
) -> dict:
    """Export data to CSV format.

    Args:
        data_type: Type of data to export
        include_archived: Whether to include archived records

    Returns:
        Export metadata
    """
    return {
        "data_type": data_type,
        "include_archived": include_archived,
        "record_count": 1234,  # Fake count for demo
        "exported_at": datetime.now().isoformat(),
    }


# =============================================================================
# Communication Functions (Communication Group)
# =============================================================================

@ui_config(
    title="Send Email",
    group="Communication",
    icon="mail",
)
@field_config(
    to_email=get_field_config('email'),
    subject=RjsfFieldConfig(placeholder="Enter email subject"),
    body=get_field_config('long_text'),
    priority=RjsfFieldConfig(
        widget='select',
        enum=['low', 'normal', 'high', 'urgent'],
    ),
)
@with_dependencies(
    FieldDependency(
        source_field='priority',
        target_field='send_immediately',
        condition=lambda v: v in ['high', 'urgent'],
        action=DependencyAction.SHOW,
    )
)
def send_email(
    to_email: str,
    subject: str,
    body: str,
    priority: str = 'normal',
    send_immediately: bool = False,
) -> dict:
    """Send an email message.

    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body content
        priority: Priority level
        send_immediately: Whether to send immediately (shown for high/urgent priority)

    Returns:
        Email sending confirmation
    """
    return {
        "to": to_email,
        "subject": subject,
        "body_length": len(body),
        "priority": priority,
        "send_immediately": send_immediately,
        "sent_at": datetime.now().isoformat(),
        "message_id": f"msg_{hash(to_email + subject) % 100000}",
    }


@group("Communication")
@with_example(
    recipient="John Doe",
    message="Your order has been shipped!",
    send_sms=True,
)
def send_notification(
    recipient: str,
    message: str,
    send_email: bool = True,
    send_sms: bool = False,
    send_push: bool = False,
) -> dict:
    """Send a multi-channel notification.

    Args:
        recipient: Name of the recipient
        message: Notification message
        send_email: Send via email
        send_sms: Send via SMS
        send_push: Send push notification

    Returns:
        Notification delivery status
    """
    channels = []
    if send_email:
        channels.append('email')
    if send_sms:
        channels.append('sms')
    if send_push:
        channels.append('push')

    return {
        "recipient": recipient,
        "message": message,
        "channels": channels,
        "sent_at": datetime.now().isoformat(),
    }


# =============================================================================
# Utility Functions (Ungrouped)
# =============================================================================

def calculate_statistics(numbers: list[float]) -> dict:
    """Calculate basic statistics for a list of numbers.

    Args:
        numbers: List of numbers

    Returns:
        Dictionary with statistical measures
    """
    if not numbers:
        return {"error": "Empty list provided"}

    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


# =============================================================================
# Main Application Setup
# =============================================================================

if __name__ == '__main__':
    # Create function groups
    admin_group = FunctionGroup(
        name="Admin",
        funcs=[create_user, delete_user],
        description="User administration functions",
        icon="shield",
        order=1,
    )

    reports_group = FunctionGroup(
        name="Reports",
        funcs=[generate_report, export_data],
        description="Reporting and data export functions",
        icon="file-text",
        order=2,
    )

    communication_group = FunctionGroup(
        name="Communication",
        funcs=[send_email, send_notification],
        description="Email and notification functions",
        icon="mail",
        order=3,
    )

    utilities_group = FunctionGroup(
        name="Utilities",
        funcs=[calculate_statistics],
        description="Utility functions",
        icon="tool",
        order=4,
    )

    # Custom CSS for the app
    CUSTOM_CSS = """
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    #sidebar {
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }

    #header {
        background: rgba(255, 255, 255, 0.95);
        border-bottom: 2px solid #667eea;
    }

    #header h1 {
        color: #667eea;
    }

    .function-item {
        transition: all 0.3s ease;
    }

    .function-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transform: translateX(5px);
    }

    .function-item:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    button[type="submit"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    button[type="submit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .form-section {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    """

    # Set up custom type registry
    registry = InputTransformRegistry()

    # Register datetime types (already done by default, but showing how)
    from datetime import datetime, date

    # Create the grouped app
    app = mk_grouped_app(
        groups=[admin_group, reports_group, communication_group, utilities_group],
        page_title="Enterprise Admin Panel",
        custom_css=CUSTOM_CSS,
    )

    print("=" * 70)
    print("Full-Featured uf Application")
    print("=" * 70)
    print()
    print("Features demonstrated:")
    print("  ✓ Function grouping and organization")
    print("  ✓ UI decorators (@ui_config, @group, etc.)")
    print("  ✓ Field configurations (email, textarea, select, etc.)")
    print("  ✓ Field dependencies (conditional display)")
    print("  ✓ Example values for testing")
    print("  ✓ Custom CSS styling")
    print("  ✓ Type transformations (datetime, date)")
    print()
    print("Available groups:")
    for group in [admin_group, reports_group, communication_group, utilities_group]:
        print(f"  • {group.name}: {len(group.funcs)} functions")
    print()
    print("Starting server at http://localhost:8080")
    print("Press Ctrl+C to stop")
    print("=" * 70)

    # For bottle apps, we can call run() directly
    if hasattr(app, 'run'):
        app.run(host='localhost', port=8080, debug=True)
    else:
        print("\nFor FastAPI apps, run:")
        print("  uvicorn examples.full_featured_example:app --reload")
