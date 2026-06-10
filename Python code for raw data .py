# ============================================
# BIBLIOTECA DE IMPORTAÇÕES
# ============================================
import asyncio  # Para programação assíncrona (executar BLE em thread separada)
import threading  # Para criar threads de execução
import queue  # Para fila thread-safe de dados
import time  # Para funções de tempo
import csv  # Para escrever ficheiros CSV
from datetime import datetime  # Para timestamps

import matplotlib.pyplot as plt  # Para criar gráficos em tempo real
from bleak import BleakScanner, BleakClient  # Biblioteca para comunicação Bluetooth Low Energy (BLE)
import numpy as np  # Para operações matemáticas e arrays

# ============================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================
# UUID da característica Bluetooth que vamos ler (identificador único do serviço BLE)
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Fila thread-safe para armazenar dados recebidos do BLE (será lida pela thread principal para plotar e guardar em CSV)
data_queue = queue.Queue()

# Evento para sinalizar à thread BLE que deve parar de receber notificações
stop_event = threading.Event() 

# Nome do ficheiro CSV com timestamp para evitar sobrescrever ficheiros anteriores
csv_filename = f'adc_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Tamanho da janela para cálculo de média móvel (suaviza o gráfico) - número de amostras
MOVING_AVG_WINDOW = 20  # Pode ser ajustado para mais/menos suavização



def notification_handler(sender, data):
    """
    Callback executado quando o dispositivo BLE envia novos dados.
    
    Esta função:
    1. Descodifica os dados recebidos do BLE
    2. Cria um timestamp do momento em que foram recebidos
    3. Converte os valores para floats
    4. Coloca os dados na fila para a thread principal processar
    
    NOTA: Esta função executa na event loop do Bleak (thread separada),
    não na thread principal. Por isso usamos a fila thread-safe.
    """
    # Descodifica os dados (assumindo que vêm no formato: "-1.00, 2.50, 3.00, ...")
    # e separa os valores pela vírgula
    decoded_data = data.decode().split(',')
    timestamp = datetime.now()  # Regista o tempo exato de receção

    try:
        # Converte cada valor (string) para número flutuante
        values = [float(x) for x in decoded_data]
    except ValueError as e:
        # Se algum valor não for um número válido, pula este e regista o erro
        print(f"Skipping invalid data: {decoded_data}, error: {e} - Only one channel.py:42")
        return

    # Coloca os dados (timestamp + lista de valores) na fila para a thread principal processar
    data_queue.put((timestamp, values))


async def bleak_main():
    """
    Função assíncrona principal para comunicação BLE.
    
    Realiza:
    1. Procura de dispositivos Bluetooth disponíveis
    2. Conecta ao dispositivo com nome contendo "bluetoothterminal"
    3. Inicia as notificações BLE (callbacks quando chegam dados)
    4. Mantém a conexão aberta até receber o sinal de stop
    5. Fecha a conexão
    """
    print("Scanning for BLE devices... - Only one channel.py:55")
    # Descobre todos os dispositivos Bluetooth Low Energy disponíveis
    devices = await BleakScanner.discover()
    # Encontra o primeiro dispositivo cujo nome contém "bluetoothterminal"
    device = next((d for d in devices if d.name and "bluetoothterminal" in d.name.lower()), None)

    if not device:
        # Se nenhum dispositivo foi encontrado, sai
        print("Device not found. Exiting BLE thread. - Only one channel.py:60")
        return

    print(f"Found device: {device.name} ({device.address}). Attempting to connect... - Only one channel.py:63")
    # Cria uma conexão com o dispositivo BLE
    async with BleakClient(device.address) as client:
        print(f"Connected to {device.name}! - Only one channel.py:65")

        # Inicia notificações: cada vez que chegam dados, chama notification_handler
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Notifications started. Waiting for stop_event... - Only one channel.py:69")

        # Mantém a conexão aberta enquanto não receber sinal de paragem
        while not stop_event.is_set():
            await asyncio.sleep(0.1)  # Pequena pausa para evitar uso excessivo de CPU

        # Ao receber sinal de paragem, desativa as notificações
        await client.stop_notify(CHARACTERISTIC_UUID)
        print("Stopped notifications. - Only one channel.py:77")

    print("BLE thread exiting. - Only one channel.py:79")


def run_bleak_loop():
    """
    Função de alvo para executar em thread separada.
    Permite que a comunicação BLE assíncrona funcione sem bloquear a thread principal.
    """
    # Inicia um novo event loop assíncrono nesta thread e executa bleak_main
    asyncio.run(bleak_main())


def smooth_with_moving_average(channel_vals: np.ndarray, window: int) -> np.ndarray:
    """
    Suaviza dados usando média móvel.
    
    A média móvel funciona calculando a média de um pequeno "janela" de valores.
    Isto remove ruído e torna o gráfico mais legível.
    
    Args:
        channel_vals: Array com os valores a suavizar
        window: Tamanho da janela (quantos valores para calcular cada média)
    
    Returns:
        Array suavizado com o mesmo tamanho que channel_vals
    """
    n = len(channel_vals)
    if window <= 1 or n < window:
        # Se a janela é muito pequena ou não há dados suficientes, retorna dados originais
        return channel_vals

    # Calcula preenchimento (padding) nos extremos para manter tamanho original
    total_pad = window - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    # Preenche os extremos do array replicando os primeiros/últimos valores
    padded = np.pad(channel_vals, (pad_left, pad_right), mode='edge')
    # Cria o kernel (máscara) da média móvel - cada valor tem peso 1/window
    kernel = np.ones(window) / window

    # Convolução: desliza a janela sobre os dados calculando a média
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


if __name__ == "__main__":
    # ============================================
    # INICIALIZAÇÃO - Inicia thread BLE
    # ============================================
    # Cria thread separada para comunicação BLE
    ble_thread = threading.Thread(target=run_bleak_loop, daemon=True)
    ble_thread.start()

    # ============================================
    # CONFIGURAÇÃO DE CANAIS A VISUALIZAR
    # ============================================
    # Especifica índices dos canais a EXCLUIR (0-indexed)
    exclude_channels = {0, 6, 7, 10, 11, 12, 13, 14, 15, 16}

    # ============================================
    # CONFIGURAÇÃO DO GRÁFICO EM TEMPO REAL
    # ============================================
    # Ativa modo interativo (actualiza gráfico em tempo real)
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Variáveis para armazenar linhas e metadados do gráfico
    lines = []
    num_channels = 0
    plot_channels = []  # Índices dos canais a plotar

    # Buffer para armazenar últimos dados
    data_buffer = []
    buffer_size = 2000  # Mantém até 2000 amostras em memória

    # Controlo do ficheiro CSV
    csv_header_written = False

    # Será preenchido com primeiro timestamp (para eixo X relativo)
    first_timestamp = None

    try:
        while True:
            # Processa todos os dados disponíveis na fila BLE
            while not data_queue.empty():
                timestamp, values = data_queue.get()

                # Se for o primeiro dado, descobre número de canais
                if num_channels == 0:
                    num_channels = len(values)
                    # Define canais a plotar (todos excepto os em exclude_channels)
                    plot_channels = [i for i in range(num_channels) if i not in exclude_channels]
                    print(f"Detectados {num_channels} canais. Plotando: {[i+1 for i in plot_channels]}")

                # Na primeira receção, cria linhas do gráfico
                if not lines and num_channels > 0:
                    for i in plot_channels:
                        line, = ax.plot([], [], label=f'Channel {i+1}')
                        lines.append(line)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('ADC Value')
                    ax.set_title('Real-time ADC Data (Moving Average)')
                    ax.legend()

                # Escreve cabeçalho CSV na primeira oportunidade
                if not csv_header_written and plot_channels:
                    with open(csv_filename, 'w', newline='') as file:
                        writer = csv.writer(file)
                        # Cabeçalho: Timestamp + canais a plotar
                        header = ['Timestamp'] + [f'Channel_{i+1}' for i in plot_channels]
                        writer.writerow(header)
                    csv_header_written = True
                    print(f"CSV criado: {csv_filename}")

                # Guarda primeiro timestamp para tempo relativo
                if first_timestamp is None:
                    first_timestamp = timestamp

                # Adiciona dados ao buffer
                data_buffer.append((timestamp, values))
                # Remove dados antigos se necessário
                if len(data_buffer) > buffer_size:
                    data_buffer.pop(0)

                # Escreve dados no CSV (modo append)
                if csv_header_written:
                    with open(csv_filename, 'a', newline='') as file:
                        writer = csv.writer(file)
                        # Apenas canais a plotar
                        row_values = [values[i] for i in plot_channels]
                        writer.writerow([timestamp] + row_values)

            # Update the plot if we have data
            if data_buffer and lines:
                # Calcula tempo relativo em segundos
                times = [(dp[0] - first_timestamp).total_seconds() for dp in data_buffer]
                times_arr = np.array(times)

                # Actualiza cada linha com dados suavizados
                for line, ch_idx in zip(lines, plot_channels):
                    channel_vals = np.array([dp[1][ch_idx] for dp in data_buffer], dtype=float)
                    smoothed = smooth_with_moving_average(channel_vals, MOVING_AVG_WINDOW)
                    line.set_data(times_arr, smoothed)

                ax.relim()
                ax.autoscale_view()

            # Redesenha a figura
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Se fechar a janela, sai
            if not plt.fignum_exists(fig.number):
                break

            # Pausa para evitar CPU excessivo
            time.sleep(0.01)

    except KeyboardInterrupt:
        # Se o utilizador pressionar Ctrl+C, mostra mensagem
        print("Keyboard interrupt received. Exiting... - Only one channel.py:213")

    finally:
        # ============================================
        # LIMPEZA E ENCERRAMENTO
        # ============================================
        # Sinaliza à thread BLE que deve parar de receber notificações
        stop_event.set()
        # Aguarda que a thread BLE termine a sua execução
        ble_thread.join()

        # Fecha a janela do gráfico
        plt.close(fig)
        print("Programa terminado com sucesso. - Only one channel.py:222")
