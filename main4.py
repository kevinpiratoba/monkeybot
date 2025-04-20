import tkinter as tk
from tkinter import ttk
from iqoptionapi.stable_api import IQ_Option
import time
import threading
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import deque
import pickle
import os
import math

class MonkeyBot:
    def __init__(self, master):
        self.master = master
        master.title("Monkey Bot Pro")
        master.geometry("500x650")
        
        # Configuración de estilo
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'))
        self.style.configure('TRadiobutton', font=('Arial', 9))
        
        # Frame principal
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Título con emoji de mono
        title_frame = ttk.Frame(self.main_frame)
        title_frame.pack(fill=tk.X, pady=5)
        ttk.Label(title_frame, text="🐵 Monkey Bot Pro 🐵", font=('Arial', 14, 'bold')).pack()
        
        # Credenciales
        cred_frame = ttk.LabelFrame(self.main_frame, text="Credenciales IQ Option")
        cred_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(cred_frame, text="Correo:").grid(row=0, column=0, sticky=tk.W)
        self.email = ttk.Entry(cred_frame)
        self.email.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        ttk.Label(cred_frame, text="Contraseña:").grid(row=1, column=0, sticky=tk.W)
        self.password = ttk.Entry(cred_frame, show="*")
        self.password.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Tipo de cuenta
        account_frame = ttk.LabelFrame(self.main_frame, text="Tipo de Cuenta")
        account_frame.pack(fill=tk.X, pady=5)
        
        self.account_type = tk.StringVar(value="PRACTICE")
        ttk.Radiobutton(account_frame, text="Demo", variable=self.account_type, value="PRACTICE").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(account_frame, text="Real", variable=self.account_type, value="REAL").grid(row=0, column=1, sticky=tk.W)
        
        # Configuración de trading
        trade_frame = ttk.LabelFrame(self.main_frame, text="Configuración de Trading")
        trade_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(trade_frame, text="Capital Inicial:").grid(row=0, column=0, sticky=tk.W)
        self.capital_inicial = ttk.Entry(trade_frame)
        self.capital_inicial.insert(0, "10")
        self.capital_inicial.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        ttk.Label(trade_frame, text="Capital Objetivo:").grid(row=1, column=0, sticky=tk.W)
        self.capital_final = ttk.Entry(trade_frame)
        self.capital_final.insert(0, "1000")
        self.capital_final.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Tamaño de vela (1 o 2 minutos)
        ttk.Label(trade_frame, text="Tamaño de Vela (min):").grid(row=2, column=0, sticky=tk.W)
        self.vela_size = ttk.Combobox(trade_frame, values=["1", "2", "5"], state="readonly")
        self.vela_size.set("1")
        self.vela_size.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Tiempo de operación
        ttk.Label(trade_frame, text="Tiempo Operación (min):").grid(row=3, column=0, sticky=tk.W)
        self.expiration_time = ttk.Combobox(trade_frame, values=["1", "2", "3", "Auto"], state="readonly")
        self.expiration_time.set("Auto")
        self.expiration_time.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Botones
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(btn_frame, text="Iniciar Monkey Bot", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(btn_frame, text="Detener Monkey Bot", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Estado y resultados
        status_frame = ttk.LabelFrame(self.main_frame, text="Estado y Resultados")
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status = ttk.Label(status_frame, text="Esperando inicio...", foreground="blue")
        self.status.pack(pady=5)
        
        self.resultado = ttk.Label(status_frame, text="Ganadas: 0 | Perdidas: 0 | Ratio: 0%")
        self.resultado.pack(pady=5)
        
        self.learning_status = ttk.Label(status_frame, text="Modelo: No entrenado")
        self.learning_status.pack(pady=5)
        
        self.current_capital = ttk.Label(status_frame, text="Capital Actual: $0.00")
        self.current_capital.pack(pady=5)
        
        self.volatility_status = ttk.Label(status_frame, text="Volatilidad: -")
        self.volatility_status.pack(pady=5)
        
        # Variables de estado
        self.win = 0
        self.loss = 0
        self.api = None
        self.activos_validos = []
        self.running = False
        self.model = None
        self.scaler = StandardScaler()
        self.history = deque(maxlen=100)
        self.model_file = "monkey_bot_model.pkl"
        self.scaler_file = "monkey_bot_scaler.pkl"
        self.current_balance = 0
        
        # Cargar modelo si existe
        self.load_model()

    def calculate_sma(self, data, window):
        """Calcula la media móvil simple"""
        if len(data) < window:
            return None
        return sum(data[-window:]) / window
    
    def calculate_ema(self, data, window):
        """Calcula la media móvil exponencial"""
        if len(data) < window:
            return None
        ema = [sum(data[:window]) / window]
        multiplier = 2 / (window + 1)
        
        for i in range(window, len(data)):
            ema_val = (data[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_val)
        
        return ema[-1]
    
    def calculate_macd(self, closes, fast=8, slow=17, signal=9):
        """Calcula el MACD manualmente"""
        if len(closes) < slow + signal:
            return None, None, None
        
        ema_fast = []
        ema_slow = []
        macd_line = []
        signal_line = []
        histogram = []
        
        for i in range(len(closes)):
            if i >= fast - 1:
                ema_f = self.calculate_ema(closes[:i+1], fast)
                ema_fast.append(ema_f)
            
            if i >= slow - 1:
                ema_s = self.calculate_ema(closes[:i+1], slow)
                ema_slow.append(ema_s)
            
            if i >= slow - 1 and (i - (slow - fast)) >= 0:
                macd_val = ema_fast[i - (slow - fast)] - ema_slow[-1]
                macd_line.append(macd_val)
                
                if len(macd_line) >= signal:
                    signal_val = self.calculate_ema(macd_line[-signal:], signal)
                    signal_line.append(signal_val)
                    
                    if len(signal_line) > 0:
                        hist_val = macd_line[-1] - signal_line[-1]
                        histogram.append(hist_val)
        
        if len(macd_line) == 0 or len(signal_line) == 0 or len(histogram) == 0:
            return None, None, None
            
        return macd_line[-1], signal_line[-1], histogram[-1]
    
    def calculate_rsi(self, closes, period=14):
        """Calcula el RSI manualmente"""
        if len(closes) < period + 1:
            return None
        
        deltas = np.diff(closes)
        seed = deltas[:period + 1]
        
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100
        if up == 0:
            return 0
            
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        for i in range(period + 1, len(deltas)):
            delta = deltas[i]
            
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta
                
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            
            if down == 0:
                rsi = 100
            elif up == 0:
                rsi = 0
            else:
                rs = up / down
                rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def load_model(self):
        """Carga el modelo y scaler desde archivos si existen"""
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.learning_status.config(text="Modelo: Cargado de archivo")
            except Exception as e:
                print(f"Error cargando modelo: {e}")
                self.model = LogisticRegression(max_iter=1000)
                self.learning_status.config(text="Modelo: Nuevo (error al cargar)")
        else:
            self.model = LogisticRegression(max_iter=1000)
            self.learning_status.config(text="Modelo: Nuevo")

    def save_model(self):
        """Guarda el modelo y scaler a archivos"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            print(f"Error guardando modelo: {e}")

    def start_bot(self):
        """Inicia el bot en un hilo separado"""
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            thread = threading.Thread(target=self.login_and_run)
            thread.start()

    def stop_bot(self):
        """Detiene el bot"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status.config(text="Bot detenido", foreground="orange")

    def login_and_run(self):
        """Método principal que maneja la conexión y ejecución del bot"""
        correo = self.email.get()
        clave = self.password.get()
        cuenta = self.account_type.get()
        
        try:
            capital = float(self.capital_inicial.get())
            objetivo = float(self.capital_final.get())
            vela_size = int(self.vela_size.get())
        except ValueError:
            self.status.config(text="Error: Valores numéricos inválidos", foreground="red")
            self.stop_bot()
            return
        
        self.api = IQ_Option(correo, clave)
        self.status.config(text="Conectando...", foreground="black")
        conectado, razon = self.api.connect()
        
        if not conectado or not self.api.check_connect():
            self.status.config(text=f"Error: {razon}", foreground="red")
            self.stop_bot()
            return
        
        self.api.change_balance(cuenta)
        self.status.config(text="Conectado. Buscando activos...", foreground="green")
        
        # Obtener activos disponibles
        activos = self.api.get_all_open_time()
        disponibles_api = self.api.get_all_ACTIVES_OPCODE()
        self.activos_validos = []
        
        for tipo in ["binary", "binary-otc"]:
            for par, datos in activos[tipo].items():
                if datos["open"] and par in disponibles_api:
                    self.activos_validos.append(par)
        
        if not self.activos_validos:
            self.status.config(text="No hay pares activos disponibles", foreground="red")
            self.stop_bot()
            return
        
        self.status.config(text=f"{len(self.activos_validos)} pares activos detectados", foreground="green")
        self.current_balance = float(self.api.get_balance())
        self.current_capital.config(text=f"Capital Actual: ${self.current_balance:.2f}")
        
        # Bucle principal de trading
        while self.running and self.current_balance <= objetivo:
            for par in self.activos_validos:
                if not self.running:
                    break
                
                # Verificar conexión
                if not self.api.check_connect():
                    self.status.config(text="Reconectando...", foreground="orange")
                    self.api.connect()
                    if not self.api.check_connect():
                        self.status.config(text="Error: reconexión fallida", foreground="red")
                        self.stop_bot()
                        return
                
                # Obtener velas (1, 2 o 5 minutos según selección)
                velas = self.api.get_candles(par, 60 * vela_size, 100, time.time())
                if not velas or len(velas) < 50:
                    continue
                
                # Analizar estrategia
                signal, features, expiration = self.analizar_estrategia(velas)
                
                if signal:
                    # Usar modelo de ML para predecir si la operación será ganadora
                    prediction = 1  # Por defecto operar (1 = operar, 0 = no operar)
                    if self.model and features:
                        try:
                            features_scaled = self.scaler.transform([features])
                            prediction = self.model.predict(features_scaled)[0]
                            proba = self.model.predict_proba(features_scaled)[0][1]
                            self.learning_status.config(text=f"Modelo: Precisión {proba*100:.1f}%")
                        except Exception as e:
                            print(f"Error en predicción: {e}")
                    
                    if prediction == 1:
                        self.status.config(text=f"{par}: Señal {signal.upper()} ({expiration} min)")
                        ok, id = self.api.buy(capital, par, signal, expiration)
                        
                        if ok:
                            resultado = self.api.check_win_v3(id)
                            
                            # Registrar resultado para aprendizaje
                            if features:
                                self.history.append((features, 1 if resultado > 0 else 0))
                                # Entrenar modelo cada 10 operaciones
                                if len(self.history) % 10 == 0 and len(self.history) > 20:
                                    self.train_model()
                            
                            if resultado > 0:
                                self.win += 1
                                self.current_balance = float(self.api.get_balance())
                                self.status.config(text=f"{par}: GANADA (+{resultado})", foreground="green")
                            else:
                                self.loss += 1
                                # Reiniciar con capital inicial después de pérdida
                                capital = float(self.capital_inicial.get())
                                self.current_balance = float(self.api.get_balance())
                                self.status.config(text=f"{par}: PERDIDA (Reiniciando capital)", foreground="red")
                            
                            ratio = self.win / (self.win + self.loss) * 100 if (self.win + self.loss) > 0 else 0
                            self.resultado.config(text=f"Ganadas: {self.win} | Perdidas: {self.loss} | Ratio: {ratio:.1f}%")
                            self.current_capital.config(text=f"Capital Actual: ${self.current_balance:.2f}")
                            time.sleep(2)
                            break
                time.sleep(1)
            time.sleep(3)
        
        if self.current_balance >= objetivo:
            self.status.config(text=f"Objetivo alcanzado! Capital: {self.current_balance}", foreground="green")
        self.stop_bot()

    def train_model(self):
        """Entrena el modelo con el historial de operaciones"""
        if len(self.history) < 20:
            return
        
        X = np.array([x[0] for x in self.history])
        y = np.array([x[1] for x in self.history])
        
        try:
            # Escalar características
            if not hasattr(self.scaler, 'mean_'):  # Si el scaler no está ajustado
                self.scaler.fit(X)
            
            X_scaled = self.scaler.transform(X)
            
            # Entrenar modelo
            self.model.fit(X_scaled, y)
            
            # Calcular precisión
            score = self.model.score(X_scaled, y)
            self.learning_status.config(text=f"Modelo: Precisión {score*100:.1f}%")
            
            # Guardar modelo
            self.save_model()
        except Exception as e:
            print(f"Error entrenando modelo: {e}")

    def analizar_estrategia(self, velas):
        """Analiza las velas y devuelve señal (call/put/None), características para ML y tiempo óptimo de expiración"""
        # Convertir velas a arrays numpy
        closes = np.array([x['close'] for x in velas])
        opens = np.array([x['open'] for x in velas])
        highs = np.array([x['max'] for x in velas])
        lows = np.array([x['min'] for x in velas])
        
        if len(closes) < 50:
            return None, None, None
        
        # Calcular indicadores manualmente
        macd, macd_signal, macd_hist = self.calculate_macd(closes, fast=8, slow=17, signal=9)
        rsi = self.calculate_rsi(closes, period=14)
        sma_9 = self.calculate_sma(closes, 9)
        sma_21 = self.calculate_sma(closes, 21)
        
        # Fractales de 30 periodos
        fractal_high = highs[-30:].max() if len(highs) >= 30 else None
        fractal_low = lows[-30:].min() if len(lows) >= 30 else None
        
        # Velas recientes
        v1 = velas[-1]
        v2 = velas[-2]
        v3 = velas[-3]
        
        # Verificar que todos los indicadores se calcularon correctamente
        if None in [macd, macd_signal, macd_hist, rsi, sma_9, sma_21, fractal_high, fractal_low]:
            return None, None, None
        
        # Calcular volatilidad (rango de las últimas 5 velas / precio actual)
        volatilidad = (highs[-5:].max() - lows[-5:].min()) / closes[-1]
        
        # Determinar tiempo de expiración basado en volatilidad
        if self.expiration_time.get() == "Auto":
            if volatilidad > 0.01:  # Mercado muy volátil
                expiration = 1  # 1 minuto para operaciones rápidas
                volatility_text = "Alta"
            elif volatilidad > 0.005:  # Mercado moderadamente volátil
                expiration = 2  # 2 minutos
                volatility_text = "Media"
            else:  # Mercado poco volátil
                expiration = 3  # 3 minutos
                volatility_text = "Baja"
            
            self.volatility_status.config(text=f"Volatilidad: {volatility_text} ({volatilidad*100:.2f}%)")
        else:
            expiration = int(self.expiration_time.get())
            self.volatility_status.config(text=f"Volatilidad: {volatilidad*100:.2f}% (Tiempo fijo)")
        
        # Condiciones para CALL
        call_cond = (
            macd > macd_signal and  # MACD cruzó al alza
            rsi > 50 and  # RSI en zona alcista
            fractal_high > sma_21 and  # Fractal por encima de SMA21
            sma_9 > sma_21 and  # Tendencia alcista
            v1['close'] > v1['open'] and  # Vela alcista
            v2['close'] > v2['open'] and  # Vela alcista previa
            macd_hist > 0  # Histograma MACD positivo
        )
        
        # Condiciones para PUT
        put_cond = (
            macd < macd_signal and  # MACD cruzó a la baja
            rsi < 50 and  # RSI en zona bajista
            fractal_low < sma_21 and  # Fractal por debajo de SMA21
            sma_9 < sma_21 and  # Tendencia bajista
            v1['close'] < v1['open'] and  # Vela bajista
            v2['close'] < v2['open'] and  # Vela bajista previa
            macd_hist < 0  # Histograma MACD negativo
        )
        
        # Características para el modelo de ML
        features = [
            macd - macd_signal,  # Diferencia MACD
            rsi,  # Valor RSI
            sma_9 - sma_21,  # Diferencia entre medias
            fractal_high - closes[-1],  # Distancia al fractal alto
            closes[-1] - fractal_low,  # Distancia al fractal bajo
            macd_hist,  # Valor del histograma MACD
            volatilidad,  # Volatilidad reciente
            sum(1 for v in [v1, v2, v3] if v['close'] > v['open']),  # Velas alcistas consecutivas
            closes[-1] - closes[-5],  # Cambio de precio en 5 velas
            expiration  # Tiempo de expiración seleccionado
        ]
        
        if call_cond:
            return "call", features, expiration
        elif put_cond:
            return "put", features, expiration
        return None, None, expiration

if __name__ == "__main__":
    root = tk.Tk()
    app = MonkeyBot(root)
    root.mainloop()