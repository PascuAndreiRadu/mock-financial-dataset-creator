import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

SIGNAL_TYPES = ['sin', 'saw_tooth', 'damped_wave']
PHASES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]


class DatasetGenerator:
    """Generates synthetic OHLCV datasets from various waveforms.
    Each dataset has four phase-shifted variants of the same signal
    mapped to close/high/low/open columns, plus a volume column.
    """

    def __init__(self, df_len=1000, noise_range=None):
        self.df_len = df_len
        self.noise_range = noise_range if noise_range is not None else (0.95, 1.05)

    def dataset_creator(self, signal_type: str = 'sin') -> pd.DataFrame:
        """Build a clean OHLCV dataframe for a given signal type.
        Args:
            signal_type: one of 'sin', 'saw_tooth', 'damped_wave'
        """
        if signal_type not in SIGNAL_TYPES:
            raise ValueError(f"Unknown signal type '{signal_type}'. Choose from {SIGNAL_TYPES}.")

        x = np.linspace(0, 4 * np.pi, self.df_len)

        match signal_type:
            case 'sin':
                y = [np.sin(x + p) for p in PHASES]
            case 'saw_tooth':
                y = [2 * ((x + p) / np.pi - np.floor((x + p) / np.pi + 0.5)) for p in PHASES]
            case 'damped_wave':
                y = [np.exp(-0.1 * x) * np.sin(x + p) for p in PHASES]

        return self._build_df(y)

    def custom_waveform_dataset(self, freq, amp, phase, sr, duration, noise=0, seed=0) -> pd.DataFrame:
        """Generate a sine-based waveform with custom parameters as an OHLCV dataframe.
        Args:
            freq:     frequency in Hz
            amp:      amplitude
            phase:    starting phase (radians). Pass None to randomise per column.
            sr:       sample rate in Hz
            duration: signal length in seconds
            noise:    std dev of gaussian noise added to the signal
            seed:     random seed for reproducibility
        """
        np.random.seed(seed)
        t = np.arange(0, duration, 1 / sr)

        signals = []
        for p in PHASES:
            ph = phase if phase is not None else np.random.uniform(0, 2 * np.pi)
            col = amp * np.sin(2 * np.pi * freq * t + ph)
            col = col + np.random.normal(0, noise, col.shape)
            signals.append(col[:self.df_len])

        return self._build_df(signals)

    def inject_noise(self, df: pd.DataFrame, noise_range: tuple | None = None) -> pd.DataFrame:
        """Multiply each column by a random uniform factor.
        Args:
            noise_range: (low, high) bounds for the multiplier.
                         Defaults to self.noise_range set at init.
        """
        if noise_range is None:
            noise_range = self.noise_range

        df = df.copy()
        for col in df.columns:
            df[col] *= np.random.uniform(noise_range[0], noise_range[1], len(df))
        return df

    def create_basics(self, folder: str):
        """Save clean and noisy versions of every signal type to disk.
        Files are saved as .pkl under the given folder path.
        Args:
            folder: directory where the files will be written
        """
        for signal in SIGNAL_TYPES:
            for add_noise in [False, True]:
                df = self.dataset_creator(signal)
                if add_noise:
                    df = self.inject_noise(df)
                df=self.add_time(df)
                label = 'noisy' if add_noise else 'clean'
                df.to_pickle(f'{folder}/{signal}_{label}.pkl')

    def _build_df(self, signals: list) -> pd.DataFrame:
        """Turn four signal arrays into a standard OHLCV dataframe."""
        return pd.DataFrame({
            'close':  signals[0],
            'high':   signals[1],
            'low':    signals[2],
            'open':   signals[3],
            'volume': np.linspace(2, 100_000, self.df_len),
        })
    
    def add_time(self,df, interval="day", start="2024-01-01", end="2026-04-03", exchange="NYSE"):
        """
        Assigns fake market timestamps to a DataFrame.
    
        intervals: "day", "hour", "30min", "15min", "5min", "1min"
        """
        freq_map = {"day": None, "hour": "1h", "30min": "30min", "15min": "15min", "5min": "5min", "1min": "1min"}
    
        cal = mcal.get_calendar(exchange)
        schedule = cal.schedule(start_date=start, end_date=end)
        freq = freq_map[interval]
    
        if freq is None:
            timestamps = pd.DatetimeIndex(schedule["market_open"].dt.tz_convert("America/New_York").values)
        else:
            bars = []
            for _, row in schedule.iterrows():
                open_  = row["market_open"].tz_convert("America/New_York")
                close_ = row["market_close"].tz_convert("America/New_York")
                session_bars = pd.date_range(open_, close_, freq=freq, tz="America/New_York")
                bars.extend(session_bars[session_bars < close_].tolist())
            timestamps = pd.DatetimeIndex(bars)
    
        # Cycle timestamps to match df length
        n = len(df)
        timestamps = timestamps.tolist()
        timestamps = (timestamps * (n // len(timestamps) + 1))[:n]
    
        result = df.copy()
        result.insert(0, "timestamp", timestamps)
        return result
        
if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.create_basics('mock_data')