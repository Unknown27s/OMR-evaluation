export interface Horse {
    name: string;
    age: number;
    breed: string;
}

export interface HorseService {
    addHorse(horse: Horse): void;
    getHorse(name: string): Horse | undefined;
    listHorses(): Horse[];
}