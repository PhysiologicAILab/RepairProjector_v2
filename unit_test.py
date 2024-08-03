import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
import numpy as np
import cv2
import logging
import torch
from PIL import Image, ImageTk
from ImageStylerApp import ImageStylerApp
import tracemalloc

# Set up logging to print to the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start monitoring memory usage
tracemalloc.start()

class TestImageStylerApp(unittest.TestCase):

    @patch('ImageStylerApp.Image.open')
    @patch('ImageStylerApp.ImageTk.PhotoImage')
    @patch('tkinter.Label')
    def setUp(self, mock_label, mock_photoimage, mock_open):
        self.root = tk.Tk()
        
        # Create a dummy image ID
        self.dummy_image_id = "dummy_image"
        
        # Mock the image opening
   
        
        # Mock PhotoImage to return the dummy image ID
        mock_photoimage.return_value = self.dummy_image_id
        
        # Mock the Label creation to use the dummy image ID
        mock_label.return_value = MagicMock(spec=tk.Label)
        
        # Patch tk.PhotoImage globally
        with patch('tkinter.PhotoImage', return_value=self.dummy_image_id):
            self.app = ImageStylerApp(self.root)
        
        logger.info('Initialized ImageStylerApp')

    def tearDown(self):
        self.root.destroy()
        logger.info('Destroyed Tk root window')
        # Print memory usage
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.reset_peak()





if __name__ == "__main__":
    unittest.main()