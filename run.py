import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

# Import your model
from Fluid_MM import MultiModalMoERec

def main():
    # Initialize configuration with your model and config file
    parameter_dict = {
        'dataset': 'micro-lens-100k-mm',  
        'config_file_list': ['config.yaml']  
    }
    
    config = Config(
        model=MultiModalMoERec, 
        dataset=parameter_dict['dataset'],
        config_file_list=parameter_dict['config_file_list']
    )
    
    # Set random seed
    init_seed(config['seed'], config['reproducibility'])
    
    # Initialize logger
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)
    
    # Create dataset
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # Prepare data
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Initialize model
    init_seed(config['seed'], config['reproducibility'])
    model = MultiModalMoERec(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # Calculate model complexity (FLOPs)
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config['device'], logger, transform)
    logger.info(set_color('FLOPs', 'blue') + f': {flops}')
    
    # Initialize trainer
    trainer = Trainer(config, model)
    
    # Train model
    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data,
        saved=True,
        show_progress=config['show_progress']
    )
    
    # Test model
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config['show_progress']
    )
    
    # Log environment and results
    environment_tb = get_environment(config)
    logger.info(
        'The running environment of this training is as follows:\n'
        + environment_tb.draw()
    )
    
    # Log validation and test results
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

if __name__ == '__main__':
    import torch
    import warnings
    warnings.filterwarnings('ignore')
    main()