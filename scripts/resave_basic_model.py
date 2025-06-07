import tensorflow as tf
from src.models.basic_localization import BasicLocalizationModel

# Load the model (this will now work because the class is registered)
model = tf.keras.models.load_model('results/basic_localization/best_model.keras', compile=False)

# Re-save the model (overwrites the old file)
model.save('results/basic_localization/best_model.keras')
print("Model re-saved successfully.") 