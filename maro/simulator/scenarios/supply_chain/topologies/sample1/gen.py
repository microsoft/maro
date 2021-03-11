from yaml import safe_load, safe_dump


def gen(number: int):
    config = None

    with open("config.yml", "rt") as fp:
        config = safe_load(fp)

        facilities = config["world"]["facilities"]

        exist_facilities = []

        for facility in facilities:
            exist_facilities.append(facility.copy())

        for i in range(number-1):
            # exist facilities
            for facility in exist_facilities:
                copied_f = facility.copy()

                copied_f["name"] = f"{facility['name']}_{i}"

                facilities.append(copied_f)

        with open(f"config_{number}.yml", "wt+") as ofp:
            safe_dump(config, ofp)


if __name__ == "__main__":
    gen(1000)
