import unittest
import os
from unittest.mock import patch, MagicMock
from converters.ide_parser import IDEParser
from converters.ipl_parser import IPLParser
from converters.dff_converter import DFFLoader
from converters.txd_converter import TXDLoader
from blender.obj_exporter import OBJExporter
from core.conversion_pipeline import ConversionPipeline

# Sample mock .IDE content
IDE_SAMPLE = """
objs
1234, model_house, 100.0, 0
end
"""

# Sample mock .IPL content
IPL_SAMPLE = """
inst
1234, model_house, 100.0, 200.0, 10.0, 0.0, 0.0, 0.0, 1.0
end
"""

class TestIDEParser(unittest.TestCase):
    def test_parse_valid_data(self):
        parser = IDEParser()
        result = parser.parse(IDE_SAMPLE.splitlines())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model_id, 1234)
        self.assertEqual(result[0].model_name, "model_house")

    def test_parse_empty_data(self):
        parser = IDEParser()
        result = parser.parse([])
        self.assertEqual(result, [])


class TestIPLParser(unittest.TestCase):
    def test_parse_valid_data(self):
        parser = IPLParser()
        result = parser.parse(IPL_SAMPLE.splitlines())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model_name, "model_house")
        self.assertEqual(result[0].position, (100.0, 200.0, 10.0))

    def test_parse_empty_data(self):
        parser = IPLParser()
        result = parser.parse([])
        self.assertEqual(result, [])


class TestDFFLoader(unittest.TestCase):
    @patch('converters.dff_converter.DFFLoader.load')
    def test_load_mocked_dff(self, mock_load):
        mock_load.return_value = {
            'vertices': [(0, 0, 0)],
            'faces': [(1, 2, 3)],
            'materials': ['material1']
        }

        loader = DFFLoader()
        data = loader.load("mock.dff")

        self.assertIn('vertices', data)
        self.assertIn('faces', data)
        self.assertIn('materials', data)
        self.assertEqual(data['vertices'][0], (0, 0, 0))


class TestTXDLoader(unittest.TestCase):
    @patch('converters.txd_converter.TXDLoader.load')
    def test_load_mocked_txd(self, mock_load):
        mock_load.return_value = {
            'material1': 'texture.png'
        }

        loader = TXDLoader()
        textures = loader.load("mock.txd")

        self.assertIn('material1', textures)
        self.assertEqual(textures['material1'], 'texture.png')


class TestOBJExporter(unittest.TestCase):
    def test_export_obj_file_creation(self):
        dummy_model = {
            'vertices': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            'faces': [(1, 2, 3)],
            'materials': ['material1']
        }

        dummy_instance = MagicMock()
        dummy_instance.model_name = "model_house"
        dummy_instance.model_data = dummy_model
        dummy_instance.position = (0.0, 0.0, 0.0)
        dummy_instance.rotation = (0.0, 0.0, 0.0, 1.0)

        output_path = "/tmp/test_model.obj"
        OBJExporter.export([dummy_instance], output_path)

        self.assertTrue(os.path.isfile(output_path))

        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("v 0.0 0.0 0.0", content)
            self.assertIn("f 1 2 3", content)

        os.remove(output_path)  # Cleanup


class TestConversionPipeline(unittest.TestCase):
    @patch('core.conversion_pipeline.ConversionPipeline.load_files')
    def test_pipeline_run_mock(self, mock_load_files):
        # Setup mock IDE and IPL instances
        mock_pipeline = ConversionPipeline("/mock/maps", "/mock/img")
        mock_pipeline.ide_models = [MagicMock(model_id=1234, model_name="model_house")]
        mock_pipeline.ipl_instances = [MagicMock(model_id=1234, model_name="model_house")]
        mock_load_files.return_value = True

        try:
            mock_pipeline.build_scene()
            mock_pipeline.export_obj("/tmp/final_result.obj")
            self.assertTrue(True)  # Test passes if no exceptions
        except Exception as e:
            self.fail(f"Pipeline failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
