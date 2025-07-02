def traffic_simulator(system):
    while True:
        # Giả lập dữ liệu cảm biến
        sim_data = {
            "lane1": {
                'l': random.randint(0, 20),
                'tđ': random.uniform(0, 120),
                'm': random.uniform(10, 100),
                'v': random.uniform(5, 60),
                'g': random.randint(5, 30),
                'ambulance': random.random() < 0.01  # 1% xác suất
            },
            # ... các làn khác
        }
        system.update_sensors(sim_data)
        system.run_cycle()
        sleep(CYCLE_TIME)