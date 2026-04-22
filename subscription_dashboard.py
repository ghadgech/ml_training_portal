import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import calendar

st.set_page_config(
    page_title="AI Subscription Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "subscriptions.json"

def load_css():
    st.markdown("""
    <style>
    body { background: #0f172a; color: #f1f5f9; }
    .main { background: #0f172a; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); }

    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.3rem 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .sub-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sub-card:hover {
        border-color: #64748b;
    }

    .status-active {
        background: #052e16;
        color: #22c55e;
        border: 1px solid #166534;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-paused {
        background: #1c1917;
        color: #f59e0b;
        border: 1px solid #92400e;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-cancelled {
        background: #1f1315;
        color: #ef4444;
        border: 1px solid #991b1b;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    .renewal-alert-soon {
        background: #1c1400;
        border-left: 4px solid #f59e0b;
        border-radius: 4px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        color: #fcd34d;
        font-size: 0.85rem;
    }

    .renewal-normal {
        background: #0a1628;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        color: #93c5fd;
        font-size: 0.85rem;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

def load_subscriptions():
    """Load subscriptions from JSON file"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_subscriptions(data):
    """Save subscriptions to JSON file"""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_days_until_renewal(billing_day):
    """Calculate days until next renewal"""
    today = date.today()
    if billing_day > today.day:
        next_renewal = date(today.year, today.month, billing_day)
    else:
        if today.month == 12:
            next_renewal = date(today.year + 1, 1, min(billing_day, 31))
        else:
            next_month = today.month + 1
            next_renewal = date(today.year, next_month, min(billing_day, 31))
    return (next_renewal - today).days

def format_status(status):
    """Format status as HTML badge"""
    if status == "active":
        return '<span class="status-active">✓ Active</span>'
    elif status == "paused":
        return '<span class="status-paused">⏸ Paused</span>'
    else:
        return '<span class="status-cancelled">✗ Cancelled</span>'

def show_overview():
    """Main overview/dashboard page"""
    subs = load_subscriptions()

    if not subs:
        st.info("📊 No subscriptions yet. Go to 'My Subscriptions' to add your first one!")
        return

    active_subs = [s for s in subs if s['status'] == 'active']
    total_monthly = sum(s['cost_monthly'] for s in active_subs)
    total_annual = total_monthly * 12

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 Monthly Spend</div>
            <div class="kpi-value">${total_monthly:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📦 Active Subs</div>
            <div class="kpi-value">{len(active_subs)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        upcoming = sum(1 for s in active_subs if get_days_until_renewal(s['billing_day']) <= 7)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📅 Due This Week</div>
            <div class="kpi-value">{upcoming}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📈 Annual Cost</div>
            <div class="kpi-value">${total_annual:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💵 Cost by Provider")
        provider_costs = {}
        for s in active_subs:
            provider = s['provider']
            provider_costs[provider] = provider_costs.get(provider, 0) + s['cost_monthly']

        if provider_costs:
            fig = px.pie(
                values=list(provider_costs.values()),
                names=list(provider_costs.keys()),
                template="plotly_dark",
                hole=0.4
            )
            fig.update_traces(marker=dict(line=dict(color='#0f172a', width=2)))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Cost by Category")
        category_costs = {}
        for s in active_subs:
            cat = s.get('category', 'Other')
            category_costs[cat] = category_costs.get(cat, 0) + s['cost_monthly']

        if category_costs:
            fig = px.bar(
                x=list(category_costs.keys()),
                y=list(category_costs.values()),
                labels={'x': 'Category', 'y': 'Monthly Cost ($)'},
                template="plotly_dark",
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # Upcoming renewals
    st.markdown('<div class="section-header">📅 Upcoming Renewals</div>', unsafe_allow_html=True)
    renewals = []
    for s in active_subs:
        days = get_days_until_renewal(s['billing_day'])
        renewals.append((s, days))
    renewals.sort(key=lambda x: x[1])

    for sub, days in renewals[:10]:
        if days <= 7:
            st.markdown(f"""
            <div class="renewal-alert-soon">
                <strong>{sub['icon']} {sub['service']}</strong> • {sub['provider']}
                <br/>Renews in <strong>{days}</strong> days on the <strong>{sub['billing_day']}th</strong> • ${sub['cost_monthly']:.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="renewal-normal">
                <strong>{sub['icon']} {sub['service']}</strong> • {sub['provider']}
                <br/>Renews in <strong>{days}</strong> days on the <strong>{sub['billing_day']}th</strong> • ${sub['cost_monthly']:.2f}
            </div>
            """, unsafe_allow_html=True)

    # Subscription cards grid
    st.markdown('<div class="section-header">🎯 Your Subscriptions</div>', unsafe_allow_html=True)

    for sub in active_subs:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                days = get_days_until_renewal(sub['billing_day'])
                st.markdown(f"""
                <div class="sub-card">
                    <strong>{sub['icon']} {sub['service']}</strong><br/>
                    {format_status(sub['status'])}<br/>
                    <small style="color: #94a3b8;">{sub['provider']} • {sub.get('tier', 'Standard')}</small>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"<strong style='color: #f1f5f9;'>${sub['cost_monthly']:.2f}/mo</strong><br/><small style='color: #94a3b8;'>Renews in {days}d</small>", unsafe_allow_html=True)

            with col3:
                if st.button("✏️", key=f"edit_{sub['id']}", help="Edit"):
                    st.session_state.edit_id = sub['id']
                    st.session_state.page = "edit"
                    st.rerun()

def show_csv_import():
    """CSV import for usage data from billing pages"""
    st.markdown('<div class="section-header">📥 Import Usage Data from CSV</div>', unsafe_allow_html=True)

    subs = load_subscriptions()
    if not subs:
        st.info("Add subscriptions first before importing usage data.")
        return

    uploaded_file = st.file_uploader("Upload CSV from billing page (Anthropic, OpenAI, etc.)", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            st.write("**📋 CSV Preview:**")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                selected_sub = st.selectbox(
                    "Which subscription is this data for?",
                    options=subs,
                    format_func=lambda x: f"{x['icon']} {x['service']} ({x['provider']})",
                    key="csv_sub_select"
                )

            with col2:
                st.write("**Available columns in CSV:**")
                columns = list(df.columns)
                for col in columns:
                    st.write(f"• {col}")

            col1, col2 = st.columns(2)
            with col1:
                usage_col = st.selectbox(
                    "Which column has usage data?",
                    options=columns,
                    help="Select the column that contains usage %, tokens used, API calls, etc."
                )

            with col2:
                # Try to extract a numeric value
                if usage_col and usage_col in df.columns:
                    latest_val = df[usage_col].iloc[-1] if len(df) > 0 else 0
                    st.write(f"**Latest value:** {latest_val}")

                    # Try to convert to percentage
                    try:
                        if isinstance(latest_val, str):
                            usage_num = float(latest_val.strip('%')) if '%' in str(latest_val) else float(latest_val)
                        else:
                            usage_num = float(latest_val)

                        # Cap at 100%
                        usage_num = min(100, max(0, usage_num))
                        st.write(f"**Parsed as:** {usage_num}%")
                    except:
                        usage_num = 0

            if st.button("✅ Update Subscription", use_container_width=True):
                try:
                    if isinstance(latest_val, str):
                        usage_num = float(latest_val.strip('%')) if '%' in str(latest_val) else float(latest_val)
                    else:
                        usage_num = float(latest_val)
                    usage_num = min(100, max(0, usage_num))

                    # Update subscription
                    for s in subs:
                        if s['id'] == selected_sub['id']:
                            s['usage_percent'] = int(usage_num)
                            s['notes'] = f"Updated {date.today().isoformat()} from CSV import"
                            break

                    save_subscriptions(subs)
                    st.success(f"✓ Updated {selected_sub['service']} usage to {int(usage_num)}%!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing usage data: {e}")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

def show_subscriptions():
    """Subscriptions management page"""
    st.subheader("📋 My Subscriptions")

    subs = load_subscriptions()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add New Subscription", use_container_width=True):
            st.session_state.page = "add"
            st.rerun()
    with col2:
        if st.button("📥 Import CSV Data", use_container_width=True):
            st.session_state.page = "csv_import"
            st.rerun()

    if not subs:
        st.info("No subscriptions yet. Click 'Add New Subscription' to get started!")
        return

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect("Filter by Status", ["active", "paused", "cancelled"], default=["active"])
    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            list(set(s.get('category', 'Other') for s in subs)),
            default=None
        )

    # Apply filters
    filtered = subs
    if status_filter:
        filtered = [s for s in filtered if s['status'] in status_filter]
    if category_filter:
        filtered = [s for s in filtered if s.get('category', 'Other') in category_filter]

    # Display as table
    if filtered:
        df_display = pd.DataFrame([{
            '': s['icon'],
            'Service': s['service'],
            'Provider': s['provider'],
            'Category': s.get('category', 'N/A'),
            'Cost': f"${s['cost_monthly']:.2f}/mo",
            'Status': s['status'].upper(),
            'Renews': f"{get_days_until_renewal(s['billing_day'])}d",
            'Actions': f"id:{s['id']}"
        } for s in filtered])

        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No subscriptions match your filters.")

    # Detailed view
    st.markdown("---")
    st.markdown('<div class="section-header">📄 Details</div>', unsafe_allow_html=True)

    for sub in filtered[:5]:  # Show first 5 in detail
        with st.expander(f"{sub['icon']} {sub['service']} ({sub['provider']})"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Tier:** {sub.get('tier', 'Standard')}")
                st.write(f"**Status:** {sub['status']}")
                st.write(f"**Monthly Cost:** ${sub['cost_monthly']:.2f}")
                st.write(f"**Billing Day:** {sub['billing_day']}th of month")

            with col2:
                st.write(f"**Start Date:** {sub.get('start_date', 'N/A')}")
                st.write(f"**Category:** {sub.get('category', 'N/A')}")
                st.write(f"**Usage:** {sub.get('usage_percent', 0)}%")
                days = get_days_until_renewal(sub['billing_day'])
                st.write(f"**Next Renewal:** {days} days")

            if sub.get('features'):
                st.write(f"**Features:** {', '.join(sub['features'])}")

            if sub.get('notes'):
                st.write(f"**Notes:** {sub['notes']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✏️ Edit", key=f"edit_detail_{sub['id']}"):
                    st.session_state.edit_id = sub['id']
                    st.session_state.page = "edit"
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{sub['id']}"):
                    subs_new = [s for s in subs if s['id'] != sub['id']]
                    save_subscriptions(subs_new)
                    st.success("✓ Subscription deleted!")
                    st.rerun()

def show_add_edit():
    """Add/Edit subscription form"""
    subs = load_subscriptions()

    edit_id = st.session_state.get('edit_id')
    if edit_id:
        sub = next((s for s in subs if s['id'] == edit_id), None)
        st.subheader(f"✏️ Edit: {sub['service']}")
        is_edit = True
    else:
        st.subheader("➕ Add New Subscription")
        sub = {
            'id': str(uuid.uuid4()),
            'service': '',
            'provider': '',
            'category': '',
            'tier': '',
            'cost_monthly': 0.0,
            'currency': 'USD',
            'billing_day': 1,
            'billing_cycle': 'monthly',
            'start_date': date.today().isoformat(),
            'status': 'active',
            'usage_percent': 0,
            'color': '#3b82f6',
            'icon': '🤖',
            'notes': '',
            'features': [],
            'website': ''
        }
        is_edit = False

    with st.form("subscription_form"):
        col1, col2 = st.columns(2)
        with col1:
            service = st.text_input("Service Name", value=sub['service'], placeholder="e.g., ChatGPT Plus")
            provider = st.text_input("Provider", value=sub['provider'], placeholder="e.g., OpenAI")

        with col2:
            category = st.selectbox(
                "Category",
                ["LLM Chat", "Creative AI", "Productivity AI", "Developer Tools", "Search AI", "Coding AI", "Other"],
                index=["LLM Chat", "Creative AI", "Productivity AI", "Developer Tools", "Search AI", "Coding AI", "Other"].index(sub.get('category', 'Other'))
            )
            tier = st.text_input("Tier/Plan", value=sub.get('tier', ''), placeholder="e.g., Plus, Pro")

        col1, col2, col3 = st.columns(3)
        with col1:
            cost = st.number_input("Monthly Cost ($)", value=sub['cost_monthly'], min_value=0.0, step=0.99)
        with col2:
            billing_day = st.number_input("Billing Day", value=sub['billing_day'], min_value=1, max_value=31, step=1)
        with col3:
            status = st.selectbox("Status", ["active", "paused", "cancelled"], index=["active", "paused", "cancelled"].index(sub['status']))

        col1, col2 = st.columns(2)
        with col1:
            usage = st.slider("Usage %", value=sub.get('usage_percent', 0), min_value=0, max_value=100, step=5)
        with col2:
            icon = st.text_input("Icon (emoji)", value=sub.get('icon', '🤖'), max_chars=2)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.fromisoformat(sub.get('start_date', date.today().isoformat())).date())
        with col2:
            website = st.text_input("Website URL", value=sub.get('website', ''), placeholder="https://...")

        notes = st.text_area("Notes", value=sub.get('notes', ''), placeholder="Any additional notes...")
        features_text = st.text_area("Features (one per line)", value='\n'.join(sub.get('features', [])), placeholder="Feature 1\nFeature 2\nFeature 3")

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("✅ Save", use_container_width=True)
        with col2:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)

        if submit:
            sub['service'] = service
            sub['provider'] = provider
            sub['category'] = category
            sub['tier'] = tier
            sub['cost_monthly'] = cost
            sub['billing_day'] = billing_day
            sub['status'] = status
            sub['usage_percent'] = usage
            sub['icon'] = icon
            sub['start_date'] = start_date.isoformat()
            sub['website'] = website
            sub['notes'] = notes
            sub['features'] = [f.strip() for f in features_text.split('\n') if f.strip()]

            if is_edit:
                subs_new = [s if s['id'] != edit_id else sub for s in subs]
                st.session_state.edit_id = None
            else:
                subs_new = subs + [sub]

            save_subscriptions(subs_new)
            st.success("✓ Subscription saved!")
            st.session_state.page = "subscriptions"
            st.rerun()

        if cancel:
            st.session_state.edit_id = None
            st.session_state.page = "subscriptions"
            st.rerun()

def show_analytics():
    """Analytics and insights page"""
    subs = load_subscriptions()
    active_subs = [s for s in subs if s['status'] == 'active']

    if not active_subs:
        st.info("No active subscriptions to analyze.")
        return

    # Time series
    st.subheader("📈 Monthly Cost Trend (12 Months)")

    today = date.today()
    months = []
    costs = []
    for i in range(-11, 1):
        d = today + timedelta(days=30*i)
        months.append(d.strftime('%b %Y'))
        costs.append(sum(s['cost_monthly'] for s in active_subs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=costs, mode='lines+markers', name='Total Monthly Cost', line=dict(color='#3b82f6', width=3)))
    fig.update_layout(template='plotly_dark', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # Cost breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💳 Cost by Provider")
        provider_costs = {}
        for s in active_subs:
            provider_costs[s['provider']] = provider_costs.get(s['provider'], 0) + s['cost_monthly']
        provider_df = pd.DataFrame(
            sorted(provider_costs.items(), key=lambda x: x[1], reverse=True),
            columns=['Provider', 'Monthly Cost']
        )
        st.dataframe(provider_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🏷️ Cost by Category")
        category_costs = {}
        for s in active_subs:
            cat = s.get('category', 'Other')
            category_costs[cat] = category_costs.get(cat, 0) + s['cost_monthly']
        category_df = pd.DataFrame(
            sorted(category_costs.items(), key=lambda x: x[1], reverse=True),
            columns=['Category', 'Monthly Cost']
        )
        st.dataframe(category_df, use_container_width=True, hide_index=True)

    # Usage analysis
    st.subheader("📊 Usage Distribution")
    usage_data = [(s['service'], s.get('usage_percent', 0)) for s in active_subs]
    usage_df = pd.DataFrame(usage_data, columns=['Service', 'Usage %'])
    fig = px.bar(usage_df, x='Service', y='Usage %', template='plotly_dark', color_discrete_sequence=['#10a37f'])
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    monthly_total = sum(s['cost_monthly'] for s in active_subs)
    with col1:
        st.metric("Monthly Spend", f"${monthly_total:.2f}")
    with col2:
        st.metric("Annual Spend", f"${monthly_total * 12:.2f}")
    with col3:
        avg_cost = monthly_total / len(active_subs) if active_subs else 0
        st.metric("Avg Cost/Service", f"${avg_cost:.2f}")
    with col4:
        st.metric("Total Services", len(active_subs))

def show_billing_calendar():
    """Billing calendar view"""
    st.subheader("📅 Billing Calendar")

    subs = load_subscriptions()
    active_subs = [s for s in subs if s['status'] == 'active']

    if not active_subs:
        st.info("No active subscriptions.")
        return

    # Group by billing day
    billing_by_day = {}
    for s in active_subs:
        day = s['billing_day']
        if day not in billing_by_day:
            billing_by_day[day] = []
        billing_by_day[day].append(s)

    # Display calendar-like view
    st.write("**Upcoming Billing Dates (Next 2 Months)**")

    for month_offset in range(2):
        today = date.today()
        if month_offset == 0:
            target_date = today
        else:
            if today.month == 12:
                target_date = date(today.year + 1, 1, 1)
            else:
                target_date = date(today.year, today.month + 1, 1)

        st.write(f"**{target_date.strftime('%B %Y')}**")

        cols = st.columns(7)
        headers = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for col, header in zip(cols, headers):
            col.write(f"**{header}**")

        # Get calendar for month
        cal = calendar.monthcalendar(target_date.year, target_date.month)

        for week in cal:
            cols = st.columns(7)
            for col, day in zip(cols, week):
                if day == 0:
                    col.write("")
                else:
                    subs_today = billing_by_day.get(day, [])
                    if subs_today:
                        total_cost = sum(s['cost_monthly'] for s in subs_today)
                        col.write(f"📅 **{day}**\n💰 ${total_cost:.0f}")
                        for s in subs_today:
                            col.write(f"• {s['icon']} {s['service']}")
                    else:
                        col.write(f"{day}")

# Main app
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'overview'
    if 'edit_id' not in st.session_state:
        st.session_state.edit_id = None

    # Sidebar navigation
    with st.sidebar:
        st.title("💳 AI Subscriptions")
        st.markdown("---")

        subs = load_subscriptions()
        active = len([s for s in subs if s['status'] == 'active'])
        monthly = sum(s['cost_monthly'] for s in active_subs := [s for s in subs if s['status'] == 'active'])

        st.metric("Active Subscriptions", active)
        st.metric("Monthly Spend", f"${monthly:.2f}")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📊 Overview", "📋 Subscriptions", "📈 Analytics", "📅 Billing Calendar"],
            key="nav"
        )

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Import", use_container_width=True):
                st.session_state.page = "import"
        with col2:
            if st.button("📤 Export", use_container_width=True):
                st.download_button(
                    label="📤",
                    data=json.dumps(subs, indent=2),
                    file_name="subscriptions.json",
                    mime="application/json"
                )

    # Page routing
    if page.startswith("📊"):
        st.session_state.page = 'overview'
    elif page.startswith("📋"):
        st.session_state.page = 'subscriptions'
    elif page.startswith("📈"):
        st.session_state.page = 'analytics'
    elif page.startswith("📅"):
        st.session_state.page = 'calendar'

    if st.session_state.page == 'overview':
        st.title("📊 AI Subscription Dashboard")
        show_overview()
    elif st.session_state.page == 'subscriptions':
        st.title("📋 My Subscriptions")
        show_subscriptions()
    elif st.session_state.page == 'csv_import':
        st.title("📥 Import Usage Data")
        show_csv_import()
    elif st.session_state.page in ('add', 'edit'):
        show_add_edit()
    elif st.session_state.page == 'analytics':
        st.title("📈 Analytics & Insights")
        show_analytics()
    elif st.session_state.page == 'calendar':
        st.title("📅 Billing Calendar")
        show_billing_calendar()

if __name__ == "__main__":
    main()
