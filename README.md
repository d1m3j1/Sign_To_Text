# SIGN LANGUAGE RECOGNITION AND TRANSLATION TO TEXT
---

## Description
---

An advanced SLR (Sign Language Recognition) system leveraging Google's MediaPipe Library for body pose landmark detection. The research contrasts LSTM and Transformer model performances, with the Transformer model showcasing notable efficacy.

###  Model
---
<table>
    <tr>
        <td>Transformer Model</td>
        <td>LSTM Model</td>
    </tr>
    <tr>
        <td><img src="visuals/transformer_model.gif" alt="Transformer Model Result"></td>
        <td><img src="visuals/lstm_model.gif" alt="LSTM Model Result"></td>
    </tr>
</table>

### LSTM Model

## How to run the code
---

First install Requirements

```
pip3 install -r requirements.txt
```

### Run the Game
--- 

```bash
python3 app.py
```

**Currently set at Transformer Model, it can be change at the [app](app.py) file**

## Conclusion
--- 

1. Our research utilized a comprehensive dataset provided by Google researchers in collaboration with the deaf community, focusing specifically on key landmarks like the hands, lips, and pose movements.
2. Two prominent models, LSTM and Transformer, were employed for the analysis, with the aim of determining the best performer for the sequence-to-sequence task.
3. The Transformer model, renowned for its multi-head attention mechanism, outshined the LSTM model, registering an impressive 58% accuracy on the test data.
4. Despite the promising results in controlled environments, both models encountered challenges when applied to real-life video scenarios.
5. The set sliding window, responsible for defining the number of frames the model uses for predictions, showcased potential drawbacks that may affect model accuracy. 
6. The Transformer model required a more extended training duration (32 hours) compared to the LSTM model (20 hours), highlighting the computational intensity of the former.

## References
---

1. To read more about the transformer model architecture, click [here](https://arxiv.org/pdf/1706.03762.pdf)

## Contributing
---

Found a bug? Create an **[issue](https://github.com/d1m3j1/sign_to_text/issues/new)**.