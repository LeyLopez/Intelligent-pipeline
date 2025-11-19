import gradio as gr
import requests
import json
from PIL import Image
from connections import chat_with_llm, predict_with_sklearn, classify_image, get_model_info


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
)


with gr.Blocks(
    theme=theme,
    title = "MLOps Final Project: Intelligent Pipeline",
    css="""
        .container {max-width: 1200px; margin: auto;}
        .header {text-align: center; padding: 20px;}
        .warning {background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;}
    """
) as demo:

    gr.Markdown(
        """
        # MLOps Final Project: Intelligent Pipeline
        This Gradio frontend connects to multiple backend services including:
        - LLM Service for conversational AI
        - SKLearn Service for Iris flower classification
        - CNN Service for image classification

        Explore each section below to interact with the respective services.
        """
    )



    with gr.Tab("LLM Chatbot"):
        gr.Markdown(
        """
        ## Chat with the LLM Service
        Enter your message below to chat with the LLM service.
        The model is connected via REST API. 
        """

    )

        chatbot = gr.Chatbot(
            height=400,
            label="LLM Chatbot"
        )   

        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                scale=4
            )
            send_btn = gr.Button("Submit", scale=1, variant="primary", scale=1)

        clear_btn = gr.Button("Clear Chat")


        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
    
            bot_message = chat_with_llm(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history
    
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        send_btn.click(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: None, None, chatbot, queue=False)




    with gr.Tab("ML Classifier (Scikit-Learn)"):
        gr.Markdown(
        """
        ## ML Classifier with Scikit-Learn
        Model trained on the Iris dataset to classify flower species based on input features.
        Enter the flower measurements below and click "Predict" to see the classification result.   
        """
    )   

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Features")

                sepal_length = gr.Slider(
                    minimum=4.0,
                    maximum=8.0,
                    value=5.1,
                    step=0.1,
                    label="Sepal Length (cm)"
                )

                sepal_width = gr.Slider(
                    minimun=2.0,
                    maximum=5.0,
                    value=3.5,
                    step=0.1,
                    label="Sepal Width (cm)"
                )

                petal_length = gr.Slider(
                    minimum=1.0,
                    maximum=7.0,
                    value=1.4,
                    step=0.1,
                    label="Petal Length (cm)"
                )

                petal_width = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    value=0.2,
                    step=0.1,
                    label="Petal Width (cm)"
                )

                predict_btn = gr.Button("Predict", variant="primary")
        
            with gr.Column():
                gr.Markdown("### Prediction Result")
                prediction_output = gr.Markdown(label="Prediction Output")

    
        predict_btn.click(
            predict_with_sklearn,
            inputs=[sepal_length, sepal_width, petal_length, petal_width],
            outputs=prediction_output
        )

        gr.Examples(
            examples=[
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 3.4, 5.4, 2.3],
                [5.9, 3.0, 4.2, 1.5]
            ],
            inputs=[sepal_length, sepal_width, petal_length, petal_width],
            label="Try These Examples"
        )   




    with gr.Tab("CNN Image Classifier"):
        gr.Markdown(
            """
            ## CNN Image Classifier
            Upload an image to classify it using the CNN model.
            The model is connected via REST API.
            This model can classify three classes: Cat, Dog, and Bird.
            """
        )

        gr.Markdown(
            """
            <div class="warning">
                 <b>Warning:</b> This model has limitted capabilities.
                 It only recognizes three classes: Cat, Dog, and Bird.
                 Any other image will likely lead to incorrect predictions.
            </div>
            """,
            elem_classes="warning"
        )

        with gr.Row():
            with gr.Column():   
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=300
                )

                classify_btn = gr.Button("Classify Image", variant="primary", size="lg")

            with gr.Column():
                classification_output = gr.Markdown(label="Classification Result")
                probability_plot = gr.Label("Class Probabilities", num_top_classes=3)


            classify_btn.click(
            classify_image,
            inputs=image_input,
            outputs=[classification_output, probability_plot]
        )



    with gr.Tab("Sistem Information"):
        gr.Markdown("## Backend Services Health Check and Capabilities")

        refresh_btn = gr.Button("Refresh Information", variant="secondary")
        info_output = gr.Markdown()

        refresh_btn.click(
            get_model_info,
            outputs=info_output
        )

        demo.load(get_model_info, outputs=info_output)



    gr.Markdown(
    """
    ### 📚 MLOps Final Project

    **Componentes:**
    - LLM Connector (Ollama/LLaMA)
    - Sklearn Model (Random Forest)
    - CNN Image (TensorFlow/Keras)
    - MLflow (Tracking & Registry)

    """
    )



if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )   
