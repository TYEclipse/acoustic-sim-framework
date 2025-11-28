"""
Unit Tests for Acoustic Simulation Framework

This module contains basic functionality tests for core components.
"""

import os
import sys
import unittest

import numpy as np

from src.array_geometry.microphone_array import MicrophoneArray
from src.data_synthesis.mixer import AudioMixer
from src.propagation_model.room_impulse_response import RoomAcousticSimulator
from src.signal_generation.noise_sources import NoiseSourceGenerator
from src.utils.labels import LabelGenerator

# 把项目根路径加入 sys.path，必须在导入 src.* 之前执行
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置固定随机种子以提高测试可重复性
np.random.seed(0)

# 抽取常量，避免文件中多处硬编码采样率
SAMPLE_RATE = 48000


class TestNoiseSourceGenerator(unittest.TestCase):
    """Test cases for NoiseSourceGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = NoiseSourceGenerator(sampling_rate=SAMPLE_RATE)
        self.duration = 1.0
        self.frequency_range = (100, 8000)

    def test_generate_engine_noise(self):
        """Test engine noise generation."""
        signal = self.generator.generate_source(
            'engine_noise',
            self.duration,
            self.frequency_range
        )

        self.assertEqual(len(signal), int(self.duration * SAMPLE_RATE))
        self.assertTrue(np.all(np.abs(signal) <= 1.0))
        self.assertGreater(np.std(signal), 0.0)

    def test_generate_wind_noise(self):
        """Test wind noise generation."""
        signal = self.generator.generate_source(
            'wind_noise',
            self.duration,
            self.frequency_range
        )

        self.assertEqual(len(signal), int(self.duration * SAMPLE_RATE))
        self.assertTrue(np.all(np.abs(signal) <= 1.0))

    def test_generate_motor_whine(self):
        """Test motor whine generation."""
        signal = self.generator.generate_source(
            'motor_whine',
            self.duration,
            (2000, 16000)
        )

        self.assertEqual(len(signal), int(self.duration * SAMPLE_RATE))
        self.assertTrue(np.all(np.abs(signal) <= 1.0))

    def test_volume_control(self):
        """Test volume control."""
        signal1 = self.generator.generate_source(
            'engine_noise',
            self.duration,
            self.frequency_range,
            volume=0.5
        )
        signal2 = self.generator.generate_source(
            'engine_noise',
            self.duration,
            self.frequency_range,
            volume=1.0
        )

        # Signal with lower volume should have lower RMS
        rms1 = np.sqrt(np.mean(signal1 ** 2))
        rms2 = np.sqrt(np.mean(signal2 ** 2))
        self.assertLess(rms1, rms2)


class TestMicrophoneArray(unittest.TestCase):
    """Test cases for MicrophoneArray."""

    def setUp(self):
        """Set up test fixtures."""
        self.positions = [
            [0.1, 0.05, 0.0],
            [0.1, -0.05, 0.0],
            [-0.1, 0.05, 0.0],
            [-0.1, -0.05, 0.0],
            [0.5, 0.8, 0.3],
            [0.5, -0.8, 0.3],
            [-1.2, 0.8, 0.2],
            [-1.2, -0.8, 0.2]
        ]
        self.array = MicrophoneArray(self.positions, name="test_array")

    def test_initialization(self):
        """Test array initialization."""
        self.assertEqual(self.array.num_mics, 8)
        self.assertEqual(self.array.name, "test_array")
        self.assertEqual(self.array.positions.shape, (8, 3))

    def test_get_mic_position(self):
        """Test getting microphone position."""
        pos = self.array.get_mic_position(0)
        np.testing.assert_array_equal(pos, [0.1, 0.05, 0.0])

    def test_cartesian_to_spherical(self):
        """Test coordinate conversion."""
        point = np.array([1.0, 0.0, 0.0])
        distance, azimuth, elevation = self.array.cartesian_to_spherical(point)

        self.assertGreater(distance, 0)
        self.assertAlmostEqual(azimuth, 0.0, places=1)
        self.assertAlmostEqual(elevation, 0.0, places=1)

    def test_spherical_to_cartesian(self):
        """Test inverse coordinate conversion."""
        distance = 1.0
        azimuth = 45.0
        elevation = 0.0

        point = self.array.spherical_to_cartesian(distance, azimuth, elevation)

        # Convert back and check
        d2, az2, el2 = self.array.cartesian_to_spherical(point)
        self.assertAlmostEqual(distance, d2, places=5)
        self.assertAlmostEqual(azimuth, az2, places=5)
        self.assertAlmostEqual(elevation, el2, places=5)

    def test_calculate_tdoa(self):
        """Test TDOA calculation."""
        source_position = np.array([1.0, 0.0, 0.0])
        tdoa = self.array.calculate_tdoa(source_position)

        self.assertEqual(len(tdoa), 8)
        self.assertAlmostEqual(tdoa[0], 0.0)  # First mic is reference


class TestRoomAcousticSimulator(unittest.TestCase):
    """Test cases for RoomAcousticSimulator."""

    def setUp(self):
        """Set up test fixtures."""
        positions = [
            [0.1, 0.05, 0.0],
            [0.1, -0.05, 0.0]
        ]
        self.mic_array = MicrophoneArray(positions)
        self.room_dims = [4.5, 1.8, 1.5]

    def test_initialization(self):
        """Test room initialization."""
        room_sim = RoomAcousticSimulator(
            self.room_dims,
            self.mic_array,
            sampling_rate=SAMPLE_RATE,
            max_order=10
        )

        self.assertIsNotNone(room_sim.room)
        self.assertEqual(room_sim.sampling_rate, SAMPLE_RATE)

    def test_add_source(self):
        """Test adding source to room."""
        room_sim = RoomAcousticSimulator(
            self.room_dims,
            self.mic_array,
            sampling_rate=SAMPLE_RATE
        )

        signal = np.random.randn(SAMPLE_RATE)
        position = np.array([1.0, 0.5, 0.5])

        source_idx = room_sim.add_source(position, signal)
        self.assertEqual(source_idx, 0)

    def test_invalid_position(self):
        """Test that invalid positions are rejected."""
        room_sim = RoomAcousticSimulator(
            self.room_dims,
            self.mic_array,
            sampling_rate=SAMPLE_RATE
        )

        signal = np.random.randn(SAMPLE_RATE)
        invalid_position = np.array([10.0, 0.0, 0.0])  # Outside room

        with self.assertRaises(ValueError):
            room_sim.add_source(invalid_position, signal)

    def test_rt60_calculation(self):
        """Test RT60 calculation."""
        room_sim = RoomAcousticSimulator(
            self.room_dims,
            self.mic_array,
            sampling_rate=SAMPLE_RATE
        )

        rt60 = room_sim.calculate_rt60()
        self.assertGreater(rt60, 0.0)
        self.assertLess(rt60, 1.0)  # Should be reasonable for vehicle cabin


class TestAudioMixer(unittest.TestCase):
    """Test cases for AudioMixer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mixer = AudioMixer(num_channels=8, sampling_rate=SAMPLE_RATE)

    def test_add_background_noise(self):
        """Test adding background noise."""
        signal = np.random.randn(8, SAMPLE_RATE) * 0.1
        noisy_signal = self.mixer.add_background_noise(
            signal,
            noise_level_db=-40,
            noise_type='pink'
        )

        self.assertEqual(noisy_signal.shape, signal.shape)
        # Noisy signal should have higher RMS
        self.assertGreater(
            np.sqrt(np.mean(noisy_signal ** 2)),
            np.sqrt(np.mean(signal ** 2))
        )

    def test_normalize_signal(self):
        """Test signal normalization."""
        signal = np.random.randn(8, SAMPLE_RATE) * 0.1
        normalized = self.mixer.normalize_signal(
            signal,
            target_level_db=-3.0,
            mode='peak'
        )

        peak = np.max(np.abs(normalized))
        target_linear = 10 ** (-3.0 / 20)
        self.assertAlmostEqual(peak, target_linear, places=5)

    def test_trim_or_pad(self):
        """Test trimming and padding."""
        signal = np.random.randn(8, SAMPLE_RATE)

        # Test trimming
        trimmed = self.mixer.trim_or_pad(signal, SAMPLE_RATE // 2)
        self.assertEqual(trimmed.shape, (8, SAMPLE_RATE // 2))

        # Test padding
        padded = self.mixer.trim_or_pad(signal, SAMPLE_RATE * 2)
        self.assertEqual(padded.shape, (8, SAMPLE_RATE * 2))


class TestLabelGenerator(unittest.TestCase):
    """Test cases for LabelGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.label_gen = LabelGenerator(format='json')

    def test_create_source_entry(self):
        """Test creating source entry."""
        entry = self.label_gen.create_source_entry(
            source_id=0,
            label='engine_noise',
            position_xyz=[1.0, 0.0, -0.5],
            clean_signal_path='/path/to/source.wav',
            orientation_az_el=[15.0, -10.0]
        )

        self.assertEqual(entry['source_id'], 0)
        self.assertEqual(entry['label'], 'engine_noise')
        self.assertEqual(len(entry['position_xyz']), 3)

    def test_validate_label(self):
        """Test label validation."""
        valid_label = {
            'clip_id': 'test_001',
            'audio_filepath': '/path/to/audio.wav',
            'sampling_rate': SAMPLE_RATE,
            'mic_array_setup': {'name': 'test'},
            'room_properties': {'dimensions': [4.5, 1.8, 1.5]},
            'sources': [
                {
                    'source_id': 0,
                    'label': 'engine_noise',
                    'position_xyz': [1.0, 0.0, 0.0]
                }
            ]
        }

        self.assertTrue(self.label_gen.validate_label(valid_label))

    def test_validate_invalid_label(self):
        """Test validation of invalid label."""
        invalid_label = {
            'clip_id': 'test_001',
            # Missing required fields
        }

        with self.assertRaises(ValueError):
            self.label_gen.validate_label(invalid_label)


def run_tests():
    """Run all tests."""
    # 使用更稳健的 argv 占位，避免空字符串造成的解析歧义
    unittest.main(argv=['first-arg-is-ignored'], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
