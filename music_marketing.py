import streamlit as st
import hashlib
from datetime import date

# ── PROSPECT POOL ──────────────────────────────────────────────────────────────────
PROSPECT_POOL = [
    # --- Cricket Influencers & Fan Pages ---
    {
        "name": "CricketKingIndia", "type": "Cricket Fan Page",
        "platform": "Instagram", "followers": "2.1M",
        "why": "Posts daily Virat Kohli content; their audience is primed for fan tributes.",
        "pitch": "Hi! We've created an AI-powered tribute song for Virat Kohli on Suno. Given your massive VK fanbase, a single reel featuring this track could explode. We'd love to share an exclusive preview — interested?",
        "icon": "🏸",
    },
    {
        "name": "ViratFansWorldwide", "type": "Fan Community",
        "platform": "YouTube", "followers": "890K",
        "why": "Dedicated Virat fan channel — they actively promote fan-made content.",
        "pitch": "We created a tribute song for King Kohli using Suno AI. It's catchy, emotional, and perfect for a YouTube Shorts campaign. Would you like an exclusive first listen before we go wide?",
        "icon": "👑",
    },
    {
        "name": "SportsTalkCricket", "type": "Sports Podcast",
        "platform": "Spotify / YouTube", "followers": "450K",
        "why": "Covers cricket culture, fan stories and viral moments — a song fits perfectly.",
        "pitch": "Hey! We produced an AI tribute song for Virat Kohli that's getting incredible reactions. Would you feature it on your next episode as the 'Fan Moment of the Week'?",
        "icon": "🎩",
    },
    {
        "name": "BharatCricketMemes", "type": "Meme & Viral Content Page",
        "platform": "Instagram / X (Twitter)", "followers": "3.4M",
        "why": "Viral content creators — they can turn the song into a trending audio clip.",
        "pitch": "We have a banger Virat Kohli tribute song made with Suno AI. With your reach, turning this into a viral audio trend on Reels and Shorts is completely doable. Want to collab?",
        "icon": "🔥",
    },
    {
        "name": "IPLInsider", "type": "IPL / Cricket News",
        "platform": "Twitter/X", "followers": "1.2M",
        "why": "Covers RCB and VK closely; fan songs align with their engagement content.",
        "pitch": "We have an original Virat Kohli tribute song created using AI. Perfect for posting during match days. Can we send you the preview link?",
        "icon": "🏆",
    },
    # --- Digital Marketing & Social Media Agencies ---
    {
        "name": "ViralVaultAgency", "type": "Social Media Agency",
        "platform": "Multi-platform", "followers": "N/A",
        "why": "Specialises in making sports content go viral — exactly what this song needs.",
        "pitch": "We have a Virat Kohli tribute song (AI-generated on Suno) that needs a professional viral push. Would your team be interested in a performance-based collaboration to distribute this across platforms?",
        "icon": "🚀",
    },
    {
        "name": "TrendMakersDigital", "type": "Digital Marketing Agency",
        "platform": "Multi-platform", "followers": "N/A",
        "why": "Works with music and sports brands to create trending campaigns.",
        "pitch": "Hi, we've produced a unique AI-composed tribute song for Virat Kohli on Suno. We're looking for an agency partner to build a multi-platform viral campaign. Open to a quick call?",
        "icon": "📈",
    },
    {
        "name": "HashtagHeroesIndia", "type": "Influencer Marketing Agency",
        "platform": "Instagram / YouTube", "followers": "N/A",
        "why": "Manages a network of cricket influencers who can amplify the song.",
        "pitch": "We have a Virat Kohli tribute song ready to launch. Your influencer network could make this a nationwide trend. Interested in a paid distribution deal?",
        "icon": "🎯",
    },
    # --- Music Promoters & Curators ---
    {
        "name": "IndieHitsIndia", "type": "Music Curator",
        "platform": "Spotify / YouTube", "followers": "320K",
        "why": "Curates independent and AI-music — would love a sports x music crossover.",
        "pitch": "We composed a Virat Kohli tribute using Suno AI. It's a crossover of cricket passion and modern AI music. Would you feature it on your playlist or channel?",
        "icon": "🎵",
    },
    {
        "name": "BollywoodVibesOfficial", "type": "Bollywood Music Page",
        "platform": "YouTube / Instagram", "followers": "5.6M",
        "why": "Sports anthems and celebrity tributes get huge traction on this channel.",
        "pitch": "We have an exciting Virat Kohli tribute song that blends cricket energy with music. Given your audience's love for sports anthems, this could be a viral hit. Can we share it?",
        "icon": "🎬",
    },
    {
        "name": "AIMusiciansCollective", "type": "AI Music Community",
        "platform": "Discord / YouTube", "followers": "75K",
        "why": "Focuses on AI-generated music — perfect early adopter community.",
        "pitch": "We used Suno AI to create a Virat Kohli tribute song. Would your community like to see it as a case study in sports-themed AI music? Happy to do a walkthrough.",
        "icon": "🤖",
    },
    # --- Brand Managers & Sponsors ---
    {
        "name": "MRF Tyres Social Team", "type": "Brand Sponsor",
        "platform": "Multi-platform", "followers": "N/A",
        "why": "MRF is Virat's long-term bat sponsor — a tribute song aligns with their brand.",
        "pitch": "We created an AI tribute song celebrating Virat Kohli's batting legacy. As MRF's social team, you could co-brand this as a fan campaign — great brand love at zero production cost. Interested?",
        "icon": "🏷️",
    },
    {
        "name": "Puma India Marketing", "type": "Brand Sponsor",
        "platform": "Instagram / YouTube", "followers": "N/A",
        "why": "Puma partners with VK — this song could be part of a brand campaign.",
        "pitch": "Hi Puma India! We've produced a Virat Kohli tribute song with Suno AI that your fans would love. Could we explore using it as part of a social content piece or campaign?",
        "icon": "👟",
    },
    {
        "name": "Dream11 Content Team", "type": "Fantasy Sports Brand",
        "platform": "YouTube / Instagram", "followers": "N/A",
        "why": "Dream11 uses celebrity content heavily in campaigns — this song fits.",
        "pitch": "We created an original Virat Kohli tribute track using AI. For Dream11, a co-branded version around match days could drive massive engagement. Open to a quick chat?",
        "icon": "🎮",
    },
    # --- YouTube Cricket Content Creators ---
    {
        "name": "CricbuzzYT", "type": "Cricket News Channel",
        "platform": "YouTube", "followers": "7.2M",
        "why": "Posts daily cricket content — a fan tribute song segment would be new.",
        "pitch": "We made an AI-powered tribute song for Virat Kohli on Suno. A 60-second segment on your channel would introduce something completely new to cricket content. Interested?",
        "icon": "📺",
    },
    {
        "name": "WisdenIndia", "type": "Cricket Heritage Brand",
        "platform": "YouTube / Instagram", "followers": "980K",
        "why": "Covers cricket history and legends — a musical tribute aligns perfectly.",
        "pitch": "We've created a heartfelt AI tribute song to Virat Kohli's journey. Wisden India could feature it as 'Fan Corner' content. Would you like a preview?",
        "icon": "📖",
    },
    {
        "name": "Sportskeeda Cricket", "type": "Sports Media",
        "platform": "YouTube / Web", "followers": "3.1M",
        "why": "One of India's largest cricket media properties — great distribution vehicle.",
        "pitch": "We produced a Virat Kohli tribute song using Suno AI. Could Sportskeeda feature it as a fan story or viral content piece? Happy to share exclusive rights for 48 hours.",
        "icon": "📰",
    },
    # --- Instagram/Shorts Creators ---
    {
        "name": "ReelKingCricket", "type": "Reels Creator",
        "platform": "Instagram", "followers": "1.8M",
        "why": "Viral cricket Reels specialist — audio trends are their bread and butter.",
        "pitch": "We have an original Virat Kohli song that could become the next big Reels audio trend. Want to be the first creator to use it and set the trend?",
        "icon": "📱",
    },
    {
        "name": "CricketShortsDaily", "type": "YouTube Shorts Creator",
        "platform": "YouTube Shorts", "followers": "620K",
        "why": "Posts daily cricket Shorts — a trending audio could go massive here.",
        "pitch": "Hey! We composed an AI Virat Kohli tribute song. Using it as background audio for your Shorts could trigger a viral audio trend. Happy to share an exclusive preview.",
        "icon": "🎥",
    },
    # --- Radio & Podcast Hosts ---
    {
        "name": "RadioMirchi Cricket Special", "type": "Radio Station",
        "platform": "Radio / Podcast", "followers": "N/A",
        "why": "Radio Mirchi does cricket specials — an AI fan song is a unique story.",
        "pitch": "Hi Radio Mirchi! We created an AI-generated Virat Kohli tribute song — it's a great angle for a cricket segment: 'When AI becomes a fan'. Would you like to feature it?",
        "icon": "📻",
    },
    {
        "name": "TheCricketPodcast India", "type": "Cricket Podcast",
        "platform": "Spotify", "followers": "95K",
        "why": "Covers cricket culture; an AI fan tribute song is a great story topic.",
        "pitch": "We have a Virat Kohli tribute song made entirely with Suno AI. This is a great podcast topic — AI as a cricket fan. Would you like to feature it in an episode?",
        "icon": "🎧",
    },
    # --- Celebrity & Fan Clubs ---
    {
        "name": "Anushka Sharma Fan Club", "type": "Celebrity Fan Page",
        "platform": "Instagram", "followers": "2.8M",
        "why": "Strong crossover audience — Anushka + Virat content always performs well.",
        "pitch": "We created a touching AI tribute song for Virat Kohli. Given your audience's love for the Kohli family, a reel using this track could get huge engagement. Want an early listen?",
        "icon": "💫",
    },
    {
        "name": "RCBFanArmy", "type": "IPL Team Fan Club",
        "platform": "Instagram / Twitter", "followers": "4.1M",
        "why": "Largest cricket fan community; VK tributes are their staple content.",
        "pitch": "Attention RCB fans! We produced an original AI tribute song for Virat Kohli. This could be THE anthem for the fan army. Want to be the first to post it?",
        "icon": "🔴",
    },
    # --- Music Distribution Platforms ---
    {
        "name": "SunoAI Featured Artists", "type": "AI Music Platform",
        "platform": "Suno / TikTok", "followers": "N/A",
        "why": "Suno's own channels feature standout creations — great for distribution.",
        "pitch": "We created a Virat Kohli tribute song on Suno that's getting great reactions. Would you consider featuring it on Suno's official channels as a standout sports tribute?",
        "icon": "🎼",
    },
    {
        "name": "DistroKid India Partner", "type": "Music Distribution",
        "platform": "Spotify / Apple Music", "followers": "N/A",
        "why": "Distributing the song to streaming platforms amplifies reach beyond social.",
        "pitch": "We have an AI-composed Virat Kohli tribute ready for distribution. Would you handle getting it on Spotify and Apple Music so cricket fans can find it organically?",
        "icon": "🌐",
    },
    # --- Event Organisers ---
    {
        "name": "IPL Watch Party Mumbai", "type": "Event Organiser",
        "platform": "Instagram / Eventbrite", "followers": "88K",
        "why": "Match-day watch parties are perfect venues to play and promote the song.",
        "pitch": "We created a Virat Kohli tribute song perfect for watch party playlists. Can we partner to play it at your next IPL event and credit it on your social posts?",
        "icon": "🏟️",
    },
    {
        "name": "CricketFest India", "type": "Sports Festival",
        "platform": "Multi-platform", "followers": "210K",
        "why": "Sports festivals drive huge cricket community engagement.",
        "pitch": "We have an original AI-generated Virat Kohli tribute song. Playing it at CricketFest and posting the crowd reaction could create a huge viral moment. Interested?",
        "icon": "🎠",
    },
    # --- Bollywood & Crossover Celebs ---
    {
        "name": "T-Series Music", "type": "Music Label",
        "platform": "YouTube", "followers": "265M",
        "why": "T-Series regularly releases sports anthems — distribution is unmatched.",
        "pitch": "We have an AI-composed Virat Kohli tribute song that's ready for a proper release. Would T-Series be interested in distributing or co-releasing this as an independent fan track?",
        "icon": "🎤",
    },
    {
        "name": "ZeeMusicOfficial", "type": "Music Label",
        "platform": "YouTube", "followers": "98M",
        "why": "Releases sports anthems and fan content regularly.",
        "pitch": "Hi Zee Music! We created a Virat Kohli tribute song using AI that has incredible potential. Would you be open to featuring or distributing it as a fan-dedicated track?",
        "icon": "🎶",
    },
    # --- Micro-Influencers ---
    {
        "name": "CricketNerd99", "type": "Cricket Micro-Influencer",
        "platform": "Instagram", "followers": "48K",
        "why": "High engagement rate; niche fans are passionate sharers.",
        "pitch": "Hey! We made a Virat Kohli tribute song with Suno AI. Given your super-engaged cricket community, a Reel with this audio could perform really well. Want to try it?",
        "icon": "🤓",
    },
    {
        "name": "SportsNationIndia", "type": "Sports Lifestyle Creator",
        "platform": "YouTube / Instagram", "followers": "230K",
        "why": "Sports x lifestyle crossover — perfect for a music + cricket content piece.",
        "pitch": "We created an AI tribute song for Virat Kohli. For your sports lifestyle content, this could be a compelling 'Fan meets Technology' story. Interested in featuring it?",
        "icon": "⚡",
    },
]

PLATFORMS = {
    "Instagram": {
        "icon": "📸",
        "tactic": "Post as a Reel with lyrics on screen. Use trending hashtags: #ViratKohli #KingKohli #CricketSong #SunoAI #CricketReels",
        "timing": "Post during match days or after a VK milestone",
        "cta": "Ask followers: 'Tag a Virat fan who needs to hear this!'",
    },
    "YouTube": {
        "icon": "▶️",
        "tactic": "Upload as a lyric video or fan tribute video. Create a YouTube Short version. Optimise title: 'Virat Kohli Tribute Song 2025 | AI Generated | Suno'",
        "timing": "Post on Sunday evenings for maximum discovery",
        "cta": "End card: 'Subscribe for more AI Cricket music'",
    },
    "Twitter/X": {
        "icon": "🐦",
        "tactic": "Post a 60-second clip with the lyric sheet as an image. Thread explaining the Suno AI creation story.",
        "timing": "Post during live match moments for trending hashtag leverage",
        "cta": "Quote tweet with: 'When AI becomes a cricket fan 🏸🎵 RT if you love VK!'",
    },
    "TikTok": {
        "icon": "🎵",
        "tactic": "Post as a trending audio — encourage users to make their own videos with the sound. Add #ViratKohli #CricketTikTok #AIMusic",
        "timing": "Evening uploads (8–10 PM IST) for maximum reach",
        "cta": "Duet challenge: 'Sing along with us!'",
    },
    "WhatsApp / Telegram": {
        "icon": "💬",
        "tactic": "Share in cricket fan groups and channels. Send to cricket team group chats. Use the 30-second hook clip to trigger shares.",
        "timing": "Share right after a VK boundary/century moment in a match",
        "cta": "Message: 'Forward this to every cricket fan you know!'",
    },
}

WEEKLY_PLAN = [
    ("Day 1 – Mon", "Launch", "Upload the song to Suno, YouTube, and Instagram Reels simultaneously. Use the full lyric video format."),
    ("Day 2 – Tue", "Outreach Burst", "Contact today's 5 prospects. Send personalised DMs/emails with the song link. Attach a 30-sec hook clip."),
    ("Day 3 – Wed", "Hashtag Storm", "Post lyric snippets on Twitter/X with trending cricket hashtags. Engage with every comment on Day 1 post."),
    ("Day 4 – Thu", "Influencer Push", "Follow up with prospects who haven't replied. Post a behind-the-scenes 'How I made a VK song with AI' Reel."),
    ("Day 5 – Fri", "Platform Expansion", "Share to TikTok and WhatsApp cricket groups. Post 15-sec teaser on Instagram Stories with poll: 'Rate this VK anthem 🔥 or 💯?'"),
    ("Day 6 – Sat", "Community Engagement", "Host a Twitter Space / YouTube Live: 'Reacting to fan responses to our VK song'. Stitch/duet most engaged posts."),
    ("Day 7 – Sun", "Analytics & Next Wave", "Review platform analytics. Identify which audience engaged most. Plan a remix or sequel based on top comments."),
]


def _day_seed() -> int:
    today = date.today().isoformat()
    return int(hashlib.md5(today.encode()).hexdigest(), 16)


def _todays_prospects():
    seed = _day_seed()
    pool = PROSPECT_POOL[:]
    pool.sort(key=lambda p: int(hashlib.md5((p["name"] + str(seed)).encode()).hexdigest(), 16))
    return pool[:5]


def show_music_marketing():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
                padding:2rem;border-radius:16px;color:white;text-align:center;margin-bottom:2rem;">
        <div style="font-size:3rem;">🎵🏸</div>
        <h1 style="margin:0.5rem 0;font-size:2rem;">Virat Kohli Tribute Song</h1>
        <p style="margin:0;opacity:0.85;font-size:1.1rem;">
            AI-Composed on Suno &nbsp;|&nbsp; Daily Viral Marketing Engine
        </p>
        <p style="margin:0.5rem 0 0;opacity:0.7;font-size:0.9rem;">
            Auto-updated every day — no manual effort needed
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Today's 5 Prospects", "📋 Song Concept", "📆 7-Day Launch Plan", "📣 Platform Strategy"])

    # ── TAB 1: DAILY 5 PROSPECTS ─────────────────────────────────────────────────────
    with tab1:
        today_str = date.today().strftime("%A, %d %B %Y")
        st.markdown(f"### Today's Targets — {today_str}")
        st.info("These 5 prospects refresh automatically every day. Each day you get a fresh set from the pool of 30+ carefully selected contacts.", icon="🔄")

        prospects = _todays_prospects()
        for i, p in enumerate(prospects, 1):
            with st.expander(f"{p['icon']}  #{i} — **{p['name']}** · {p['type']} · {p['platform']} · {p['followers']} followers", expanded=(i == 1)):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Why they're the right fit:**")
                    st.write(p["why"])
                with col2:
                    st.markdown("**Best platform to reach them:**")
                    st.markdown(f"`{p['platform']}`")

                st.markdown("---")
                st.markdown("**📝 Ready-to-send outreach message:**")
                st.code(p["pitch"], language=None)

                st.markdown(
                    "<p style='font-size:0.8rem;color:grey;'>Copy the message above, personalise the opening line with their first name, and send via DM or email.</p>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.success("**Pro tip:** Send all 5 messages before 10 AM IST for the best response rate. Follow up 48 hours later if no reply.", icon="✅")

    # ── TAB 2: SONG CONCEPT ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🎵 Song Concept Sheet")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Title:** *King of the Crease*
*(Working title — customise to your preference)*

**Platform Created On:** Suno AI

**Genre:** Inspirational / Bollywood-Fusion Sports Anthem

**Language:** Hindi / English Mix (Hinglish)

**Duration:** ~2:30–3:00 min

**Mood:** Celebratory, Emotional, High-energy
""")
        with col2:
            st.markdown("""
**Core Theme:**
Tribute to Virat Kohli's journey from a young Delhi boy to the greatest batter of his generation.

**Target Audience:**
Cricket fans aged 15–45, RCB fans, IPL viewers, sports enthusiasts across India and the diaspora.

**Unique Angle:**
First AI-composed cricket tribute song in Hinglish — bridges AI music tech with sports fandom.

**Viral Hook:**
Chorus line that fans can sing along to during match moments.
""")

        st.markdown("---")
        st.markdown("### 🎤 Song Structure")
        song_data = {
            "Section": ["Intro (0–20s)", "Verse 1 (20–55s)", "Chorus (55–75s)", "Verse 2 (75–110s)", "Chorus x2 (110–130s)", "Bridge (130–150s)", "Final Chorus + Outro (150–180s)"],
            "Theme": [
                "Crowd noise + stadium ambiance + soft guitar build",
                "VK's early life, hunger, 2003 U-19 journey",
                "Main hook — 'King of the crease, fire in his eyes…'",
                "Test cricket grit, overseas record, chase master",
                "Repeat hook — audience sing-along moment",
                "Emotional slow-down — family, resilience, legacy",
                "Full orchestra build + fan chant ending",
            ],
        }
        st.table(song_data)

        st.markdown("---")
        st.markdown("### 💡 Suno AI Creation Tips")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("**Style Prompt:**\nBollywood-fusion sports anthem, orchestral, crowd energy, Hinglish lyrics, emotional build-up")
        with col_b:
            st.info("**Mood Tags:**\nCelebratory, Inspirational, High BPM, Anthemic, Stadium feel")
        with col_c:
            st.info("**Voice Style:**\nMale lead, powerful vocals, backing choir for chorus, crowd claps on beat")

    # ── TAB 3: 7-DAY LAUNCH PLAN ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📆 7-Day Viral Launch Plan")
        st.caption("Run this plan in the first week after publishing. You don't need to be online — schedule posts in advance using Buffer or Hootsuite.")

        for day, phase, action in WEEKLY_PLAN:
            with st.expander(f"**{day}** — {phase}"):
                st.write(action)

        st.markdown("---")
        st.markdown("### 📊 Success Metrics to Track")
        metrics = {
            "Metric": ["Total Plays", "Shares / Reposts", "Comments", "Influencer Pick-ups", "Prospects Replied", "Hashtag Reach"],
            "Target (Week 1)": ["10,000+", "500+", "200+", "3+", "2 out of 5 daily", "50,000+ impressions"],
            "Platform": ["YouTube + Suno", "Instagram + Twitter", "All platforms", "Instagram / YouTube", "DM / Email", "Twitter/X + Instagram"],
        }
        st.table(metrics)

    # ── TAB 4: PLATFORM STRATEGY ──────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📣 Platform-by-Platform Viral Strategy")

        for platform, details in PLATFORMS.items():
            with st.expander(f"{details['icon']}  **{platform}**"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Tactic**")
                    st.write(details["tactic"])
                with col2:
                    st.markdown("**Best Timing**")
                    st.write(details["timing"])
                with col3:
                    st.markdown("**Call-to-Action**")
                    st.write(details["cta"])

        st.markdown("---")
        st.markdown("### 🔑 The Golden Rule of Viral Music")
        st.warning("""
**Consistency over virality.**
Post something related to the song every single day for 30 days — even if it's just a quote, a lyric card, or a fan comment screenshot. The algorithm rewards persistent creators, not one-time posters.

You don't need one big break. You need 30 small touchpoints that compound.
        """, icon="💡")

        st.markdown("### 📬 Outreach Template (General Purpose)")
        st.code("""Subject: Virat Kohli AI Tribute Song — Collaboration Opportunity

Hi [Name],

I've created an AI-composed tribute song for Virat Kohli using Suno —
it's a Bollywood-fusion anthem celebrating his journey from Delhi streets
to becoming the greatest batter of his generation.

Given your audience's love for cricket, I believe this could create
a special moment for your community.

I'd love to share a private preview link with you before the public launch.

Would you be open to a quick listen?

Best,
[Your Name]
""", language=None)
