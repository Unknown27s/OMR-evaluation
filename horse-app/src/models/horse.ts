export class Horse {
    name: string;
    age: number;
    breed: string;

    constructor(name: string, age: number, breed: string) {
        this.name = name;
        this.age = age;
        this.breed = breed;
    }

    getDetails(): string {
        return `${this.name} is a ${this.age} year old ${this.breed}.`;
    }

    celebrateBirthday(): void {
        this.age += 1;
    }
}