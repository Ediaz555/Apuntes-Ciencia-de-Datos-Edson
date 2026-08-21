import math

import joblib
import streamlit as st


st.set_page_config(page_title="Predicción de problemas cardiacos", page_icon="❤️")

st.title("Predicción de problemas cardiacos")
st.image(
	"https://img.magnific.com/vector-gratis/estilo-dibujos-animados-corazon_78370-7988.jpg?semt=ais_hybrid&w=740&q=80",
	width=420,
)
st.write(
	"### Objetivo de la aplicación"
	"\nEsta aplicación experimental estima, a partir de la edad y el nivel de colesterol, "
	"la posible presencia de problemas cardiacos mediante una red neuronal. No sustituye "
	"la evaluación de un profesional de la salud."
)
st.info(
	"### Instrucciones de uso\n"
	"1. Introduzca una edad entre 20 y 100 años.\n"
	"2. Introduzca el colesterol entre 200 y 500.\n"
	"3. Pulse **Realizar predicción** y revise el resultado."
)


def forward(X1, X2):
	"""Compute a forward pass of the supplied neural network."""
	a1 = max(0, -0.35 - 1.2 * X1 - 1.4 * X2)
	a2 = max(0, 0.45 + 0.26 * X1 + 0.83 * X2)
	a3 = max(0, 1.0 + 1.3 * X1 + 0.98 * X2)
	a4 = max(0, 0.30 - 1.1 * X1 + 1.1 * X2)
	a5 = max(0, -0.69 + 0.50 * X1 + 0.51 * X2)
	a6 = max(0, -0.39 - 0.13 * X1 - 1.4 * X2)
	a7 = max(0, -0.11 + 1.7 * X1 + 0.010 * X2)
	a8 = max(0, 0.60 - 0.10 * X1 + 2.0 * X2)
	a9 = max(0, 0.27 + .077*a1 + .065*a2 + .021*a3 - .26*a4 + .067*a5 - .26*a6 + .34*a7 + .35*a8)
	a10 = max(0, -2.3 + .62*a1 - .36*a2 + .45*a3 + .74*a4 - .15*a5 + 1.4*a6 + .93*a7 + .066*a8)
	a11 = max(0, .91 + .38*a1 + .043*a2 + .67*a3 + .68*a4 + .45*a5 - .71*a6 - 1.4*a7 - 1.2*a8)
	a12 = max(0, .70 - 1.5*a1 + .26*a2 + 1.4*a3 + .94*a4 + .97*a5 + .25*a6 - .86*a7 - .61*a8)
	a13 = max(0, 1.2 - .34*a1 - .83*a2 + .39*a3 + .70*a4 - .095*a5 - .23*a6 - .30*a7 - 1.4*a8)
	a14 = max(0, 1.1 - .76*a1 - .37*a2 - .22*a3 - .90*a4 + .30*a5 + .88*a6 - .43*a7 - .091*a8)
	a15 = max(0, .10 - .018*a9 - .29*a10 - .41*a11 - .41*a12 - .16*a13 - .40*a14)
	a16 = max(0, .021 - .62*a9 + 1.2*a10 + 1.5*a11 + .14*a12 + .20*a13 + .90*a14)
	a17 = max(0, 2.1 - .13*a9 - 1.6*a10 - .19*a11 - .57*a12 - .21*a13 + 1.3*a14)
	a18 = max(0, -.71 - .84*a9 + 1.7*a10 + .84*a11 - 1.2*a12 + 1.4*a13 - .21*a14)
	a19 = max(0, .071 - .39*a15 - .38*a16 - .45*a17 - .086*a18)
	a20 = max(0, 1.4 + .021*a15 + 1.4*a16 - 1.8*a17 - 1.7*a18)
	return math.tanh(-1.7 + .40*a19 + 1.6*a20)


age = st.number_input("Edad (años)", min_value=20, max_value=100, value=40, step=1)
cholesterol = st.number_input("Colesterol", min_value=200, max_value=500, value=220, step=1)

if st.button("Realizar predicción", type="primary"):
	try:
		scaler = joblib.load("modelo_estandarizacion.joblib")
		normalized = scaler.transform([[age, cholesterol]])[0] * 2
	except Exception as error:
		st.error(f"No se pudo cargar modelo_estandarizacion.joblib: {error}")
		st.stop()

	score = forward(float(normalized[0]), float(normalized[1]))
	prediction = 1 if score >= 0 else -1
	percentage = (1 / (1 + math.exp(-4 * score))) * 100

	if prediction == -1:
		st.success("no sufriras del corazon")
		st.image("https://static.vecteezy.com/system/resources/previews/019/617/125/non_2x/healthy-heart-cartoon-png.png", width=300)
	else:
		st.error("sufriras de problemas de corazon")
		st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT22U_Y0ZQTmyO_x5zthmL-IIlVTen64VX7KFNqMZQZrQ&s=10", width=300)
		st.metric("Porcentaje estimado", f"{percentage:.2f}%")
		if age >= 60 or cholesterol >= 240:
			recommendation = "Consulte pronto a un profesional, controle el colesterol y mantenga actividad física segura."
		else:
			recommendation = "Mantenga una alimentación equilibrada, actividad física regular y controles médicos."
		st.warning(f"Recomendación: {recommendation}")

st.markdown("---\n**Trademark: Edson Diaz**\n\n*esto es un trabajo experimental 2026*")
