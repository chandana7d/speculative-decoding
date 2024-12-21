import streamlit as st
from speculative_decoding import SpeculativeDecoder
from main import load_models
import torch
import time

@st.cache_resource
def initialize_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main_model_name = "gpt2-medium"    #"gpt2-xl"
    guide_model_name = "gpt2-medium"
    main_model, guide_model, main_tokenizer, guide_tokenizer = load_models(main_model_name, guide_model_name, device)
    speculative_decoder = SpeculativeDecoder(main_model, guide_model, main_tokenizer=main_tokenizer, guide_tokenizer=guide_tokenizer)
    return speculative_decoder, main_model, main_tokenizer, guide_tokenizer, device

def main():
    st.title("Real-Time Token-by-Token Comparison")
    st.write("Compare speculative decoding and greedy decoding in real-time with token-by-token generation.")

    speculative_decoder, main_model, tokenizer,guide_tokenizer, device = initialize_models()

    prompt = st.text_input("Enter a prompt:", value="Once upon a time, in a distant land,")
    max_length = st.slider("Maximum output length:", min_value=10, max_value=100, value=50)

    if st.button("Generate Outputs"):
        with st.spinner("Generating outputs..."):
            # Speculative decoding
            speculative_output, speculative_time = speculative_decoder.decode_with_timing(prompt, max_length)

            # Greedy decoding
            start_time = time.time()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            greedy_output = tokenizer.decode(
                main_model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=False
                )[0], skip_special_tokens=True
            )
            greedy_time = time.time() - start_time

        # Display results
        st.subheader("Speculative Decoding")
        st.text_area("Output (Speculative)", speculative_output, height=150)
        st.write(f"Time Taken: {speculative_time:.2f} seconds")

        st.subheader("Greedy Decoding")
        st.text_area("Output (Greedy)", greedy_output, height=150)
        st.write(f"Time Taken: {greedy_time:.2f} seconds")

        # Token-by-token comparison
        st.subheader("Token-by-Token Comparison")
        speculative_tokens = speculative_output.split()
        greedy_tokens = greedy_output.split()

        comparison = ""
        for i, (spec_token, greedy_token) in enumerate(zip(speculative_tokens, greedy_tokens)):
            comparison += f"Token {i+1}: Speculative - `{spec_token}`, Greedy - `{greedy_token}`\n"

        st.text_area("Comparison", comparison, height=200)

if __name__ == "__main__":
    main()