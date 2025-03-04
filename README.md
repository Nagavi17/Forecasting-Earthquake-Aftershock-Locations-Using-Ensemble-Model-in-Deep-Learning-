# Forecasting-Earthquake-Aftershock-Locations-Using-Ensemble-Model-in-Deep-Learning-
This project uses an Ensemble Model combining LSTM, GRU, and RNN to predict aftershock locations based on seismic data. A 1000 km radius around the main shock is analyzed to capture spatiotemporal correlations. Performance is evaluated using precision, recall, and F1 score, aiding disaster response and mitigation.
Aftershocks following major earthquakes can pose significant threats to infrastructure and human life. Accurate prediction of aftershock locations is crucial for effective disaster preparedness and mitigation efforts. This project introduces a novel approach utilizing deep learning models, specifically an Ensemble Model, to predict aftershock locations after a significant earthquake event.

Methodology
The approach focuses on identifying high-magnitude earthquake epicenters, which act as gravitational centers for subsequent aftershock predictions. The deep learning models are trained using seismic data and historical earthquake trends to capture the geographic and temporal correlations between main shocks and their aftershocks.

A spatial radius of approximately 1000 kilometers around the main shock epicenter is considered, as this is where aftershocks are most likely to occur. The Ensemble Model integrates multiple advanced neural network architectures, including:

Long Short-Term Memory (LSTM)
Gated Recurrent Unit (GRU)
Recurrent Neural Networks (RNNs)
Dataset & Evaluation
The model is trained and validated using an extensive seismic dataset covering a wide range of geological regions and earthquake magnitudes over a significant time span. Performance is evaluated using key metrics such as:

Precision
Recall
F1 Score
Impact & Applications
The proposed method enhances the accuracy and efficiency of aftershock prediction, facilitating timely emergency response and evacuation strategies. By integrating deep learning techniques into seismic hazard assessment, this approach has the potential to revolutionize earthquake forecasting and reduce the impact of aftershocks on vulnerable communities and critical infrastructure.
