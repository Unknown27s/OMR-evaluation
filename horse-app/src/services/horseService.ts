import { Horse } from '../models/horse';

export class HorseService {
    private horses: Horse[] = [];

    public addHorse(horse: Horse): void {
        this.horses.push(horse);
    }

    public getHorse(name: string): Horse | undefined {
        return this.horses.find(horse => horse.name === name);
    }

    public listHorses(): Horse[] {
        return this.horses;
    }
}