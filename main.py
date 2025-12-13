from humlet_simulation.simulation import Simulation

if __name__ == "__main__":
    # Initialize the simulation with a 800x600 window and 1000 humlets
    sim = Simulation(world_width=1200, world_height=1000, num_humlets=100)

    sim.run()
