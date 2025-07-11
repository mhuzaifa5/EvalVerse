import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from llm_model import llm_summary,get_dashboard_summary,chat_with_llm

# ------------------ Model Data ------------------
model_data = {
    "Deepseek-MoE-16B": {
        "likes": {'145'},
        "downloads": {"Jan": 4500, "Feb": 5800, "Mar": 6800, "Apr": 7500, "May": 8200},
        "company": "Deepseek AI",
        "followers": "81.2K",
        "params": "16B",
        "tensor_type": ["BF16"],
        "model_format": ["safetensors"],
        "Artificial Intelligence Index": 88,
        "Input Context Length": 128000,
        "Output Context Length": 128000,
        "last_updated": "20 days ago",
        "country": "China"
    },
    "Gemma-2-27B-IT": {
        "likes": {'547'},
        "downloads": {"Jan": 3500, "Feb": 4000, "Mar": 4500, "Apr": 5000, "May": 5800},
        "company": "Google",
        "followers": "19.2K",
        "params": "27B",
        "tensor_type": ["BF16"],
        "model_format": ["safetensors"],
        "Artificial Intelligence Index": 90,
        "Input Context Length": 8192,
        "Output Context Length": 8192,
        "last_updated": "5 days ago",
        "country": "USA"
    },
    "Qwen1.5-72B-Chat": {
        "likes": {'217'},
        "downloads": {"Jan": 3000, "Feb": 3800, "Mar": 4500, "Apr": 5200, "May": 6000},
        "company": "Alibaba",
        "followers": "38K",
        "params": "72B",
        "tensor_type": ["BF16"],
        "model_format": ["safetensors"],
        "Artificial Intelligence Index": 86,
        "Input Context Length": 32768,
        "Output Context Length": 32768,
        "last_updated": "15 days ago",
        "country": "China"
    },
    "Llama-3-70B-Instruct": {
        "likes": {'1490'},
        "downloads": {"Apr": 15000, "May": 25000, "Jun": 35000},
        "company": "Meta",
        "followers": "52K",
        "params": "70B",
        "tensor_type": ["BF16"],
        "model_format": ["safetensors", "PyTorch"],
        "Artificial Intelligence Index": 95,
        "Input Context Length": 8192,
        "Output Context Length": 8192,
        "last_updated": "1 day ago",
        "country": "USA"
    },
    "Mistral-7B-Instruct-v0.3": {
        "likes": {'1880'},
        "downloads": {"Jan": 5500, "Feb": 6800, "Mar": 8000, "Apr": 9000, "May": 9800},
        "company": "Mistral AI",
        "followers": "10.3K",
        "params": "7B",
        "tensor_type": ["BF16"],
        "model_format": ["safetensors"],
        "Artificial Intelligence Index": 89,
        "Input Context Length": 32768,
        "Output Context Length": 32768,
        "last_updated": "7 days ago",
        "country": "France"
    }
}

# ------------------ Streamlit Config ------------------
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>\U0001F4CA EvalVerse - An AI-Integrated CompareHub </h1>  ", unsafe_allow_html=True)
# ------------------ Dashboard Summarizer  ------------------
options = {
    "Models": [
        "Text-to-Text (Text Generation)",  # Supported
        "Text-to-Image",
        "Image-to-Image",
        "Image-to-Video",
        "Video-to-Audio"
    ],
    "Datasets": [
        "Image Datasets",
        "Text Datasets",
        "Audio Datasets",
        "Video Datasets",
        "Table Datasets"
    ]
}

selection = st.selectbox(
    "Select entities you want to compare (model type / datasets)",
    options["Models"] + options["Datasets"]
)

# ------------------ Logic ------------------
if not selection:
    st.warning("Please select a model type or dataset to continue.")
    st.stop()

if selection == "Text-to-Text (Text Generation)":
    available_models = list(model_data.keys())
    selected_models = st.multiselect("Compare Text Generation Models", available_models, default=available_models, label_visibility="collapsed")
    
    if not selected_models:
        st.warning("Select at least one text generation model.")
        st.stop()
    
    # === Paste your full dashboard logic here ===
    # This includes charts, summaries, dataframe views etc. 

else:
    st.info(f"🚧 The feature for '{selection}' is under development. Stay tuned — we're working on adding this comparison soon!")
    st.stop()

# ------------------ Dashboard Code ------------------

# Your existing dashboard code here (charts, dataframes, etc.)

if st.button("🤖 Summarize Dashboard"):
    with st.spinner("Capturing dashboard and analyzing..."):
        summary = get_dashboard_summary(selected_models, model_data)
        with st.expander("📊 Gemini Model Summary (click to expand/collapse)", expanded=True):
            st.write(summary)
    

# ------------------ LLM Summary Button ------------------
def render_header_with_summary(title, key, selected_models):
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f"### {title}")
    with col2:
        if st.button("✨", key=key):
            prompt = (
                f"Generate a concise summary for the non technical person for {title} and {key} only after giving one line brief "
                f"what the {title} and {key} means in term of AI models (nothing else ) of selected AI models: {', '.join(selected_models)} , "
                f"also include which model is leading in particular {key}."
            )
            st.session_state[f"summary_{key}"] = llm_summary(prompt)

    if f"summary_{key}" in st.session_state:
        with st.expander("📋 View Summary", expanded=True):
            st.markdown(
                f"<div style='margin-top: -10px; font-size: 0.9em; color: gray;'>{st.session_state[f'summary_{key}']}</div>",
                unsafe_allow_html=True
            )

# ------------------ Selection ------------------


# ------------------ Data Preparation ------------------
likes_df = pd.DataFrame([{ "Model": m, "Likes": int(next(iter(model_data[m]["likes"]))) } for m in selected_models])
downloads_df = pd.DataFrame([
    {"Model": m, "Month": k, "Downloads": v}
    for m in selected_models for k, v in model_data[m]["downloads"].items()
])
params_df = pd.DataFrame([
    {"Model": m, "Parameters (B)": float(model_data[m]["params"].replace("B", "")), "Bubble Size": float(model_data[m]["params"].replace("B", ""))}
    for m in selected_models
])
country_df = pd.DataFrame([
{"Model": m, "Country": model_data[m]["country"]}
for m in selected_models
])
tensor_df = pd.DataFrame([{ "Model": m, "Tensor Types": ", ".join(model_data[m]["tensor_type"]) } for m in selected_models])
format_df = pd.DataFrame([{ "Model": m, "Formats": ", ".join(model_data[m]["model_format"]) } for m in selected_models])
update_df = pd.DataFrame([{ "Model": m, "Last Updated": model_data[m]["last_updated"] } for m in selected_models])
company_df = pd.DataFrame([{ "Model": m, "Company": model_data[m]["company"] } for m in selected_models])
follower_df = pd.DataFrame([
    { "Company": model_data[m]["company"], "Followers": float(model_data[m]["followers"].replace("K", "")) * 1000 }
    for m in selected_models
])
ai_index_df = pd.DataFrame([{ "Model": m, "AI Index": model_data[m]["Artificial Intelligence Index"] } for m in selected_models])
context_data = [(m, model_data[m]["Input Context Length"], model_data[m]["Output Context Length"]) for m in selected_models]


# ------------------ Charts ------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="bordered-column">', unsafe_allow_html=True)
    render_header_with_summary("\U0001F4C8 Downloads", "monthly downloads only", selected_models)
    fig = px.line(downloads_df, x="Month", y="Downloads", color="Model", markers=True)
    st.plotly_chart(fig, use_container_width=True, height=300)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    render_header_with_summary("❤️ Likes", "model likes only", selected_models)
    fig = px.bar(likes_df, x="Model", y="Likes", color="Model", text="Likes")
    st.plotly_chart(fig, use_container_width=True, height=300)

with col3:
    render_header_with_summary("\U0001F9EE Parameters", "llm no. of parameters (size) only", selected_models)
    fig = px.scatter(params_df, x="Model", y="Parameters (B)", size="Bubble Size", color="Model", size_max=60)
    st.plotly_chart(fig, use_container_width=True, height=300)

# ------------------ Combined Row: AI Index and Context Lengths ------------------
col4, col5 = st.columns(2)
with col4:
    render_header_with_summary("🧠 Artificial Intelligence Index", "model ai_index only ", selected_models)
    fig = px.bar(ai_index_df, x="Model", y="AI Index", color="Model", text="AI Index")
    st.plotly_chart(fig, use_container_width=True)

with col5:
    render_header_with_summary("🔄 Context Lengths by Model", "model input and output context_lengths", selected_models)
    x_input, x_output = 0, 1
    all_lengths = [val for _, i, o in context_data for val in (i, o)]
    min_len = min(all_lengths)
    max_len = max(all_lengths)

    def scale_size(length, min_size=10, max_size=50):
        return min_size + (length - min_len) / (max_len - min_len) * (max_size - min_size)

    fig = go.Figure()
    for idx, (model, input_len, output_len) in enumerate(context_data):
        y_pos = idx
        fig.add_trace(go.Scatter(
            x=[x_input],
            y=[y_pos],
            mode='markers',
            marker=dict(size=scale_size(input_len), color='skyblue', line=dict(color='steelblue', width=1)),
            name='Input Context Length' if idx == 0 else None,
            showlegend=(idx == 0),
            text=[f"{model}<br>Input: {input_len}"],
            hoverinfo="text"
        ))
        fig.add_trace(go.Scatter(
            x=[x_output],
            y=[y_pos],
            mode='markers',
            marker=dict(size=scale_size(output_len), color='orange', line=dict(color='darkorange', width=1)),
            name='Output Context Length' if idx == 0 else None,
            showlegend=(idx == 0),
            text=[f"{model}<br>Output: {output_len}"],
            hoverinfo="text"
        ))

    fig.add_trace(go.Scatter(
        x=[x_input, x_output],
        y=[-1, -1],
        mode='text',
        text=["Input Context Length", "Output Context Length"],
        textposition="bottom center",
        textfont=dict(size=14, color='black'),
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(tickvals=[x_input, x_output], ticktext=["Input", "Output"], range=[-0.6, 1.4]),
        yaxis=dict(tickvals=list(range(len(context_data))), ticktext=[m for m, _, _ in context_data], autorange='reversed', range=[len(context_data)+1, -2]),
        height=350,
        showlegend=True,
        legend=dict(x=1.05, y=1),
        margin=dict(r=250)
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Mid Row ------------------
col6, col7, col8 = st.columns(3)
with col6:
    render_header_with_summary("🔢 Tensor Types", "model tensor_types (BF16 or FP32)", selected_models)
    st.dataframe(tensor_df, use_container_width=True, height=200)

with col7:
    render_header_with_summary("💾 Model Formats", "model weights formats (safetensors , pt , ONNX etc. only)", selected_models)
    st.dataframe(format_df, use_container_width=True, height=200)

with col8:
    render_header_with_summary("🕒 Last Updated", "model last_updated only ", selected_models)
    st.dataframe(update_df, use_container_width=True, height=200)

# ------------------ Final Row ------------------
col9, col10 = st.columns(2)
with col9:
    render_header_with_summary("🏢 Companies", "model released by companies", selected_models)
    st.dataframe(company_df, use_container_width=True, height=150)

with col10:
    render_header_with_summary("👥 Company Followers", "model released by company_followers", selected_models)
    fig = px.bar(follower_df, x="Company", y="Followers", text="Followers", color="Company")
    st.plotly_chart(fig, use_container_width=True, height=250)

# ------------------ Country Highlight Map ------------------
st.markdown("### 🌍 Global Presence of Selected AI Models")

# Prepare country-wise model info
country_model_map = {}
for m in selected_models:
    country = model_data[m]["country"]
    country_model_map.setdefault(country, []).append(m)

# Create a DataFrame with country, model count, and hover text
choropleth_df = pd.DataFrame([
    {
        "Country": country,
        "Model Count": len(models),
        "Model Names": ", ".join(models)
    }
    for country, models in country_model_map.items()
])

# Create choropleth map
fig = px.choropleth(
    choropleth_df,
    locations="Country",
    locationmode="country names",
    color="Model Count",
    color_continuous_scale="Viridis",
    hover_name="Country",
    hover_data={"Model Names": True, "Model Count": True},
    title="AI Models by Country (Hover for Model Names)",
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    height=500,
    margin=dict(t=40, l=0, r=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ------------------ LLM Chat Assistant ------------------
st.markdown("### 🤖 Ask about AI Model Selection")

# Chat input box
user_message = st.text_input("Type your question about the AI models here:")

if user_message:
    with st.spinner("Thinking..."):
        reply = chat_with_llm(user_message, selected_models, model_data)

    # Display only the latest response (no chat history)
    st.markdown(f"**Model Response:** {reply}")
