import time
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state
from snake_ai_recording import SimpleModel
from snake_ai_testrunner import (  
    BreedingStrategy, FitnessFunction, SelectionMethod,
    train_population_with_strategies, ExperimentTracker
)

def train_best_model(combination_id, breeding_strategy, fitness_function, selection_method):
    population_size = 200
    mutation_rate = 0.05
    mutation_scale = 0.1
    elite_size = 50
    generations = 100

    tracker = ExperimentTracker(f"{combination_id}_training")
    log_file = tracker.base_dir / f'{combination_id}_log.txt'

    def log_message(message: str):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def training_callback(generation: int, avg_food: float, max_food: int,
                          avg_fitness: float, best_fitness: float, best_food: int):
        gen_message = f"\nGeneration {generation + 1} Stats:"
        gen_message += f"\n  Average Fitness: {avg_fitness:.2f}"
        gen_message += f"\n  Average Food: {avg_food:.2f}"
        gen_message += f"\n  Max Food This Gen: {max_food}"
        gen_message += f"\n  Best Food Ever: {best_food}"
        gen_message += f"\n  Best Fitness Overall: {best_fitness:.2f}"
        gen_message += f"\n{'-' * 40}"
        log_message(gen_message)

        tracker.save_generation_data(combination_id, generation, avg_food,
                                      max_food, avg_fitness, best_fitness, best_food)

    # Training
    start_time = time.time()
    best_model, final_best_fitness, final_best_food = train_population_with_strategies(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        elite_size=elite_size,
        breeding_strategy=breeding_strategy,
        fitness_function=fitness_function,
        selection_method=selection_method,
        callback=training_callback
    )
    training_time = time.time() - start_time

    tracker.save_experiment_result(
        combination_id=combination_id,
        config={
            'breeding_strategy': breeding_strategy.__name__,
            'fitness_function': fitness_function.__name__,
            'selection_method': selection_method.__name__,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'mutation_scale': mutation_scale,
            'elite_size': elite_size
        },
        training_time=training_time
    )

    tracker.plot_generation_data()
    tracker.save_results()

    summary_message = f"\nTraining completed for {combination_id}"
    summary_message += f"\nBest fitness: {final_best_fitness:.2f}"
    summary_message += f"\nBest food count: {final_best_food}"
    summary_message += f"\nTraining time: {training_time:.2f}s"
    log_message(summary_message)

# Virker kun med en kombination af gangen
if __name__ == "__main__":
    
    
    # Combination 1 DEN BEDSTE
    train_best_model(
        combination_id="Buniform_crossover_Fbasic_fitness_Stournament_selection",
        breeding_strategy=BreedingStrategy.uniform_crossover,
        fitness_function=FitnessFunction.basic_fitness,
        selection_method=SelectionMethod.tournament_selection
    )
    
    '''
    # Combination 2 TREDJE BEDST
    train_best_model(
        combination_id="Buniform_crossover_Fexploration_focused_fitness_Stournament_selection",
        breeding_strategy=BreedingStrategy.uniform_crossover,
        fitness_function=FitnessFunction.exploration_focused_fitness,
        selection_method=SelectionMethod.tournament_selection
    )
    
    # Combination 3 ANDEN BEDST
    train_best_model(
        combination_id="Btwo_point_crossover_Fbasic_fitness_Stournament_selection",
        breeding_strategy=BreedingStrategy.two_point_crossover,
        fitness_function=FitnessFunction.basic_fitness,
        selection_method=SelectionMethod.tournament_selection
    )
    '''