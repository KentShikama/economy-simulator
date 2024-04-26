# Minimal simulator of an economy

## Description
Simulates an economy of three cities with a total of five people. The economy consists of two raw materials (water and fertilizer) and one consumable good (apples). One person produces the water in city A, another person produces the fertilizer in city B, and a third person combines the water and the fertilizer to produce apples in city C. The two remaining people are peddlers that transport the goods between the cities. On each day, each person has a chance of performing one action depending on their role.

## How to run
```bash
python3 main.py
```

## Example output (first 100 days)

Day 0
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $99   Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 11} Action: Digger bought apple from Carrier X for 1.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 11} Action: Dirt bought apple from Carrier Y for 1.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 10} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $100  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 10} Action: Carrier X bought apple from Digger for 0.95.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $101  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Carrier Y moved to A.

Day 1
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $98   Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 11} Action: Digger bought apple from Carrier X for 0.9974999999999999.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 11} Action: Dirt produced 10 units of fertilizer.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 10} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $101  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $101  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Carrier Y moved to C.

Day 2
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $98   Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 11} Action: Digger refuses to buy apple from Carrier X because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 11} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Farmer Joe sold apple to Carrier Y for 1.05.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $101  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $99   Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 10} Action: Carrier Y moved to B.

Day 3
Digger has a chance of consuming an apple.
City: A  Money: $98   Fullness: 96  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 10, 'fertilizer': 0, 'apple': 11} Action: Digger collected 10 units of water.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 10} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 96  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $101  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $99   Fullness: 96  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 11} Action: Carrier Y bought apple from Dirt for 0.95.

Day 4
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $98   Fullness: 95  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 10, 'fertilizer': 0, 'apple': 11} Action: Digger refuses to buy apple from Carrier X because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 10} Action: Dirt bought apple from Carrier Y for 0.9476249999999999.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $101  Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.05'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $99   Fullness: 95  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 10} Action: Carrier Y moved to A.

Day 5
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 94  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 10, 'fertilizer': 0, 'apple': 12} Action: Digger bought apple from Carrier Y for 0.99500625.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 10} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Farmer Joe refuses to sell apple to Carrier X because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $101  Fullness: 96  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier X refuses to sell apple to Farmer Joe because the price is too low.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $99   Fullness: 94  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 9} Action: Carrier Y bought water from Digger for 0.95.

Day 6
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $98   Fullness: 93  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 12} Action: Digger refuses to sell water to Carrier Y because the price is too low.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 10} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $101  Fullness: 95  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier X moved to B.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $99   Fullness: 93  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 9} Action: Carrier Y refuses to sell apple to Digger because the price is too low.

Day 7
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 92  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.94'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 13} Action: Digger bought apple from Carrier Y for 0.9925187343749999.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 9} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $101  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $100  Fullness: 92  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 8} Action: Carrier Y moved to B.

Day 8
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 91  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.94'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 13} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 10, 'apple': 9} Action: Dirt refuses to buy apple from Carrier X because the price is too high.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 6} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $100  Fullness: 99  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 7} Action: Carrier X bought water from Carrier Y for 0.95.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $101  Fullness: 91  Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier Y refuses to sell apple to Dirt because the price is too low.

Day 9
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 90  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.94'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 13} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 9} Action: Dirt sold fertilizer to Carrier X for 1.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 6} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $99   Fullness: 98  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 7} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $102  Fullness: 90  Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Carrier Y sold apple to Dirt for 0.99500625.

Day 10
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 89  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 13} Action: Digger refuses to buy apple from Carrier X because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 9} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: B  Money: $99   Fullness: 97  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 7} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $102  Fullness: 89  Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Carrier Y refuses to sell apple to Carrier X because the price is too low.

Day 11
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 12} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1.05', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 9} Action: Dirt sold fertilizer to Carrier Y for 1.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: C  Money: $99   Fullness: 96  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 7} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of consuming an apple.
City: C  Money: $101  Fullness: 88  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 7} Action: Carrier Y moved to C.

Day 12
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 99  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 12} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1.05', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 9} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 4} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $99   Fullness: 95  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 7} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $102  Fullness: 87  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Carrier Y sold fertilizer to Farmer Joe for 1.

Day 13
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 98  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 12} Action: Digger refuses to buy apple from Carrier X because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1.05', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 8} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 3} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $100  Fullness: 94  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 6} Action: Carrier X sold apple to Digger for 1.0395393094160157.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $102  Fullness: 86  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Carrier Y moved to B.

Day 14
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 97  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 13} Action: Digger refuses to buy apple from Carrier X because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 8} Action: Dirt refuses to sell fertilizer to Carrier Y because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 2} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $100  Fullness: 93  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 6} Action: Carrier X refuses to sell apple to Digger because the price is too low.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $101  Fullness: 85  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.94'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier Y bought apple from Dirt for 0.9452559374999999.

Day 15
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 12} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$0.94'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 8} Action: Dirt bought apple from Carrier Y for 0.9381842267479539.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 2} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $100  Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 5} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $101  Fullness: 84  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.94'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Carrier Y bought apple from Dirt for 0.9428927976562499.

Day 16
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$1.04'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 11} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$0.94'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 8} Action: Dirt bought apple from Carrier Y for 0.9358387661810841.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 1} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $100  Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 4} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $102  Fullness: 83  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Carrier Y moved to A.

Day 17
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 99  Prices: {'water': '$0.95', 'fertilizer': '$1', 'apple': '$0.99'} Inventory: {'water': 9, 'fertilizer': 0, 'apple': 12} Action: Digger bought apple from Carrier Y for 0.9826307044901383.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$0.94'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 7} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $99   Fullness: 99  Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 4} Action: Carrier X bought water from Digger for 0.9476249999999999.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $103  Fullness: 82  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 6} Action: Carrier Y refuses to sell apple to Carrier X because the price is too low.

Day 18
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$0.94'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 13} Action: Digger bought apple from Carrier Y for 0.980174127728913.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$0.94'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 6} Action: Dirt consumed apple.
City: C  Money: $100  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: A  Money: $100  Fullness: 98  Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 3} Action: Carrier X sold apple to Carrier Y for 1.0291828341153586.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $103  Fullness: 81  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 6} Action: Carrier Y moved to B.

Day 19
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$0.98'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 13} Action: Digger refuses to buy apple from Carrier X because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 6} Action: Dirt refuses to buy apple from Carrier Y because the price is too high.
City: C  Money: $100  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: B  Money: $100  Fullness: 97  Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 3} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $104  Fullness: 80  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Carrier Y sold apple to Carrier X for 1.0421446710937499.

Day 20
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 96  Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$0.98'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 13} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $100  Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 6} Action: Dirt refuses to buy apple from Carrier X because the price is too high.
City: C  Money: $100  Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $100  Fullness: 96  Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 3} Action: Carrier X sold apple to Dirt for 1.0369404611424755.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $105  Fullness: 79  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 4} Action: Carrier Y sold apple to Carrier X for 1.0395393094160155.

Day 21
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$0.98'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 12} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $98   Fullness: 97  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 7} Action: Dirt refuses to buy apple from Carrier X because the price is too high.
City: C  Money: $100  Fullness: 96  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $100  Fullness: 95  Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 3} Action: Carrier X sold apple to Carrier Y for 1.0779403708815738.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $104  Fullness: 78  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Carrier Y moved to A.

Day 22
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$1.03'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 12} Action: Digger refuses to buy apple from Carrier Y because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $98   Fullness: 96  Prices: {'water': '$1', 'fertilizer': '$1.0', 'apple': '$1.09'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 7} Action: Dirt refuses to buy apple from Carrier X because the price is too high.
City: C  Money: $100  Fullness: 95  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $101  Fullness: 94  Prices: {'water': '$0.9', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 3} Action: Carrier X sold fertilizer to Dirt for 0.9950062499999999.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $104  Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 4} Action: Carrier Y consumed apple.

Day 23
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$1.03'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 11} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $96   Fullness: 95  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 8} Action: Dirt bought apple from Carrier X for 1.0369404611424755.
City: C  Money: $100  Fullness: 94  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $101  Fullness: 93  Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 2} Action: Carrier X bought fertilizer from Dirt for 0.9452559374999998.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $104  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 3} Action: Carrier Y consumed apple.

Day 24
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1', 'apple': '$1.03'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 10} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 94  Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 8} Action: Dirt refuses to buy apple from Carrier X because the price is too high.
City: C  Money: $100  Fullness: 93  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $101  Fullness: 92  Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 2} Action: Carrier X moved to C.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $104  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 3} Action: Carrier Y refuses to buy apple from Digger because the price is too high.

Day 25
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 99  Prices: {'water': '$1.04', 'fertilizer': '$1', 'apple': '$1.03'} Inventory: {'water': 7, 'fertilizer': 0, 'apple': 10} Action: Digger sold water to Carrier Y for 0.9974999999999999.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 93  Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 8} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
City: C  Money: $100  Fullness: 92  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$0.9', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 1} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of consuming an apple.
City: A  Money: $102  Fullness: 98  Prices: {'water': '$0.95', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 4} Action: Carrier Y bought apple from Digger for 1.0317622397146453.

Day 26
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 100 Prices: {'water': '$1.04', 'fertilizer': '$1', 'apple': '$1.08'} Inventory: {'water': 7, 'fertilizer': 0, 'apple': 8} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 7} Action: Dirt consumed apple.
City: C  Money: $100  Fullness: 91  Prices: {'water': '$1', 'fertilizer': '$0.95', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of consuming an apple.
City: C  Money: $102  Fullness: 99  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 1} Action: Carrier X sold water to Farmer Joe for 1.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $103  Fullness: 97  Prices: {'water': '$0.95', 'fertilizer': '$1.0', 'apple': '$1.07'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 3} Action: Carrier Y sold apple to Digger for 1.0833503517003775.

Day 27
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 100 Prices: {'water': '$1.04', 'fertilizer': '$1', 'apple': '$1.03'} Inventory: {'water': 7, 'fertilizer': 0, 'apple': 8} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 6} Action: Dirt consumed apple.
Farmer Joe has a chance of growing apples.
City: C  Money: $99   Fullness: 90  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 10} Action: Farmer Joe grew 10 units of apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: C  Money: $102  Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of consuming an apple.
City: A  Money: $102  Fullness: 96  Prices: {'water': '$0.95', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 4} Action: Carrier Y bought apple from Digger for 1.0291828341153586.

Day 28
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 100 Prices: {'water': '$1.04', 'fertilizer': '$1', 'apple': '$1.08'} Inventory: {'water': 7, 'fertilizer': 0, 'apple': 6} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 5} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $102  Fullness: 99  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $102  Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 3} Action: Carrier Y consumed apple.

Day 29
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $97   Fullness: 100 Prices: {'water': '$1.04', 'fertilizer': '$1', 'apple': '$1.08'} Inventory: {'water': 7, 'fertilizer': 0, 'apple': 5} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 5} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $102  Fullness: 98  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $103  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 3} Action: Carrier Y sold water to Digger for 1.0447565625.

Day 30
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$1', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 4} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 4} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: A  Money: $102  Fullness: 97  Prices: {'water': '$0.95', 'fertilizer': '$0.95', 'apple': '$1.09'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $102  Fullness: 98  Prices: {'water': '$0.95', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 3} Action: Carrier Y bought water from Carrier X for 0.947625.

Day 31
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$1', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 0, 'apple': 3} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 3} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $104  Fullness: 96  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.09'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier X sold fertilizer to Digger for 1.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $102  Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 3} Action: Carrier Y refuses to buy water from Digger because the price is too high.

Day 32
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $94   Fullness: 99  Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.03'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 4} Action: Digger bought apple from Carrier Y for 1.0189295358467596.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 2} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.95'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 8} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: C  Money: $104  Fullness: 95  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.09'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $103  Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.07'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 1} Action: Carrier Y consumed apple.

Day 33
Digger has a chance of selling water.
Digger has a chance of buying an apple.
Digger has a chance of consuming an apple.
City: A  Money: $94   Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 4} Action: Digger refuses to buy apple from Carrier Y because the price is too high.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 2} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 7} Action: Farmer Joe sold apple to Carrier X for 1.0887874841995993.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $103  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $103  Fullness: 99  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.07'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 1} Action: Carrier Y moved to C.

Day 34
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $94   Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 4} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $97   Fullness: 100 Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 1} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 6} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
City: A  Money: $103  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of consuming an apple.
City: B  Money: $103  Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.07'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 1} Action: Carrier Y moved to B.

Day 35
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $94   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 3} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $96   Fullness: 99  Prices: {'water': '$1', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 2} Action: Dirt bought apple from Carrier Y for 1.0698760126390976.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: B  Money: $103  Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
City: B  Money: $104  Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.07'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 1} Action: Carrier Y bought apple from Dirt for 1.0291828341153586.

Day 36
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $94   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 2} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $98   Fullness: 98  Prices: {'water': '$1', 'fertilizer': '$1.04', 'apple': '$1.08'} Inventory: {'water': 0, 'fertilizer': 7, 'apple': 1} Action: Dirt sold fertilizer to Carrier Y for 0.9974999999999999.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: B  Money: $102  Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Carrier X bought fertilizer from Carrier Y for 0.9476249999999999.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $104  Fullness: 96  Prices: {'water': '$1.04', 'fertilizer': '$1.0', 'apple': '$1.07'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 1} Action: Carrier Y sold water to Dirt for 1.

Day 37
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $94   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 1} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $96   Fullness: 97  Prices: {'water': '$0.95', 'fertilizer': '$1.04', 'apple': '$1.03'} Inventory: {'water': 1, 'fertilizer': 7, 'apple': 2} Action: Dirt bought apple from Carrier Y for 1.0672013226075.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 96  Prices: {'water': '$1.0', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier X sold fertilizer to Dirt for 1.0421446710937499.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying apple.
City: B  Money: $105  Fullness: 95  Prices: {'water': '$1.04', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 1} Action: Carrier Y bought apple from Dirt for 1.0266098770300702.

Day 38
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $94   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 0} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
Dirt has a chance of consuming an apple.
City: B  Money: $95   Fullness: 96  Prices: {'water': '$0.95', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 8, 'apple': 2} Action: Dirt bought apple from Carrier Y for 1.0645333193009812.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.0'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 5} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
City: A  Money: $103  Fullness: 95  Prices: {'water': '$1.0', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying apple.
City: C  Money: $106  Fullness: 94  Prices: {'water': '$1.04', 'fertilizer': '$1.0', 'apple': '$1.12'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier Y moved to C.

Day 39
Digger has a chance of selling water.
City: A  Money: $95   Fullness: 99  Prices: {'water': '$1.04', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 7, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier X for 0.9950062500000001.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $95   Fullness: 95  Prices: {'water': '$0.95', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 8, 'apple': 2} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 96  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 4} Action: Farmer Joe sold apple to Carrier Y for 1.1177599852660303.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: A  Money: $101  Fullness: 94  Prices: {'water': '$0.95', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier X bought fertilizer from Digger for 0.95.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $104  Fullness: 100 Prices: {'water': '$1.04', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Carrier Y consumed apple.

Day 40
Digger has a chance of selling water.
City: A  Money: $96   Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.08'} Inventory: {'water': 7, 'fertilizer': 0, 'apple': 0} Action: Digger refuses to sell water to Carrier X because the price is too low.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $95   Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 8, 'apple': 1} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 3} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $102  Fullness: 93  Prices: {'water': '$0.95', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 0} Action: Carrier X sold fertilizer to Digger for 0.9974999999999999.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
City: C  Money: $103  Fullness: 99  Prices: {'water': '$1.04', 'fertilizer': '$1.0', 'apple': '$1.01'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 1} Action: Carrier Y bought apple from Farmer Joe for 1.0447565625.

Day 41
Digger has a chance of selling water.
City: A  Money: $95   Fullness: 97  Prices: {'water': '$0.94', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 7, 'fertilizer': 1, 'apple': 0} Action: Digger refuses to sell water to Carrier X because the price is too low.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $95   Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 8, 'apple': 0} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 2} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: B  Money: $102  Fullness: 92  Prices: {'water': '$0.95', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $103  Fullness: 98  Prices: {'water': '$1.04', 'fertilizer': '$1.0', 'apple': '$1.01'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 1} Action: Carrier Y moved to A.

Day 42
Digger has a chance of selling water.
Digger has a chance of buying an apple.
City: A  Money: $96   Fullness: 96  Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.08'} Inventory: {'water': 6, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier Y for 1.0421446710937499.
Dirt has a chance of selling fertilizer.
City: B  Money: $96   Fullness: 99  Prices: {'water': '$0.95', 'fertilizer': '$1.04', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 7, 'apple': 0} Action: Dirt sold fertilizer to Carrier X for 0.9900374375390625.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 2} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $101  Fullness: 91  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier X refuses to buy water from Dirt because the price is too high.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: A  Money: $103  Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 0} Action: Carrier Y sold apple to Digger for 1.0779403708815738.

Day 43
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $96   Fullness: 95  Prices: {'water': '$1.04', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 1} Action: Digger sold water to Carrier Y for 0.9900374375390624.
Dirt has a chance of selling fertilizer.
City: B  Money: $96   Fullness: 98  Prices: {'water': '$0.95', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 7, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 1} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 90  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X bought water from Dirt for 0.95.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
City: A  Money: $104  Fullness: 96  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 0} Action: Carrier Y sold water to Digger for 1.0369404611424755.

Day 44
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $95   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 6, 'fertilizer': 1, 'apple': 0} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
City: B  Money: $97   Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$0.94', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 7, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 1} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: A  Money: $100  Fullness: 89  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
City: A  Money: $103  Fullness: 95  Prices: {'water': '$0.94', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier Y bought water from Carrier X for 0.94289279765625.

Day 45
Digger has a chance of selling water.
City: A  Money: $96   Fullness: 99  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier X for 0.9900374375390626.
Dirt has a chance of selling fertilizer.
City: B  Money: $97   Fullness: 96  Prices: {'water': '$1.0', 'fertilizer': '$0.94', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 7, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $100  Fullness: 88  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: C  Money: $103  Fullness: 94  Prices: {'water': '$0.94', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier Y moved to C.

Day 46
Digger has a chance of selling water.
City: A  Money: $96   Fullness: 98  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $97   Fullness: 95  Prices: {'water': '$1.0', 'fertilizer': '$0.94', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 7, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
City: C  Money: $102  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $101  Fullness: 87  Prices: {'water': '$0.94', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X sold fertilizer to Farmer Joe for 0.9974999999999999.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying water.
City: A  Money: $103  Fullness: 93  Prices: {'water': '$0.94', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier Y moved to A.

Day 47
Digger has a chance of selling water.
City: A  Money: $96   Fullness: 97  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 0} Action: Digger refuses to sell water to Carrier Y because the price is too low.
Dirt has a chance of selling fertilizer.
City: B  Money: $97   Fullness: 94  Prices: {'water': '$1.0', 'fertilizer': '$0.94', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 7, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
City: C  Money: $101  Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: A  Money: $101  Fullness: 86  Prices: {'water': '$0.94', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
City: A  Money: $103  Fullness: 92  Prices: {'water': '$0.99', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier Y refuses to buy water from Carrier X because the price is too high.

Day 48
Digger has a chance of selling water.
City: A  Money: $97   Fullness: 96  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 4, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier Y for 0.9850934380853517.
Dirt has a chance of selling fertilizer.
City: B  Money: $97   Fullness: 93  Prices: {'water': '$1.0', 'fertilizer': '$0.94', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 7, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
City: C  Money: $101  Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling water.
City: C  Money: $101  Fullness: 85  Prices: {'water': '$0.94', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: B  Money: $102  Fullness: 91  Prices: {'water': '$0.94', 'fertilizer': '$1.0', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 0, 'apple': 0} Action: Carrier Y moved to B.

Day 49
Digger has a chance of selling water.
City: A  Money: $97   Fullness: 95  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 4, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $98   Fullness: 92  Prices: {'water': '$1.0', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 6, 'apple': 0} Action: Dirt sold fertilizer to Carrier Y for 0.99500625.
City: C  Money: $101  Fullness: 96  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: A  Money: $101  Fullness: 84  Prices: {'water': '$0.94', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: B  Money: $102  Fullness: 90  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y sold water to Dirt for 0.9974999999999999.

Day 50
Digger has a chance of selling water.
City: A  Money: $97   Fullness: 94  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 4, 'fertilizer': 1, 'apple': 0} Action: Digger refuses to sell water to Carrier X because the price is too low.
Dirt has a chance of selling fertilizer.
City: B  Money: $97   Fullness: 91  Prices: {'water': '$0.95', 'fertilizer': '$0.94', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 6, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier Y because the price is too low.
City: C  Money: $101  Fullness: 95  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
City: C  Money: $101  Fullness: 83  Prices: {'water': '$0.94', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
City: B  Money: $101  Fullness: 89  Prices: {'water': '$0.98', 'fertilizer': '$0.9', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 2, 'apple': 0} Action: Carrier Y bought fertilizer from Dirt for 0.9358387661810841.

Day 51
Digger has a chance of selling water.
City: A  Money: $97   Fullness: 93  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 4, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $98   Fullness: 90  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 5, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier Y because the price is too low.
City: C  Money: $101  Fullness: 94  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of selling water.
City: B  Money: $101  Fullness: 82  Prices: {'water': '$0.94', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $101  Fullness: 88  Prices: {'water': '$0.98', 'fertilizer': '$0.9', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to A.

Day 52
Digger has a chance of selling water.
City: A  Money: $98   Fullness: 92  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier Y for 0.9826307044901383.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 89  Prices: {'water': '$0.95', 'fertilizer': '$0.98', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt sold fertilizer to Carrier X for 0.9875623439452148.
City: C  Money: $101  Fullness: 93  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 81  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X refuses to buy water from Dirt because the price is too high.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: B  Money: $100  Fullness: 87  Prices: {'water': '$0.93', 'fertilizer': '$0.9', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to B.

Day 53
Digger has a chance of selling water.
City: A  Money: $98   Fullness: 91  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 88  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
City: C  Money: $101  Fullness: 92  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: A  Money: $100  Fullness: 80  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
City: C  Money: $100  Fullness: 86  Prices: {'water': '$0.93', 'fertilizer': '$0.9', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to C.

Day 54
Digger has a chance of selling water.
City: A  Money: $98   Fullness: 90  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Digger refuses to sell water to Carrier X because the price is too low.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 87  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
City: C  Money: $101  Fullness: 91  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Farmer Joe has no apple to consume.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $100  Fullness: 79  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $101  Fullness: 85  Prices: {'water': '$0.98', 'fertilizer': '$0.9', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 2, 'apple': 0} Action: Carrier Y sold water to Farmer Joe for 0.9974999999999999.

Day 55
Digger has a chance of selling water.
City: A  Money: $98   Fullness: 89  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 86  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of growing apples.
City: C  Money: $100  Fullness: 90  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 10} Action: Farmer Joe grew 10 units of apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $100  Fullness: 78  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $100  Fullness: 84  Prices: {'water': '$0.98', 'fertilizer': '$0.9', 'apple': '$1.01'} Inventory: {'water': 2, 'fertilizer': 2, 'apple': 1} Action: Carrier Y bought apple from Farmer Joe for 0.9900374375390624.

Day 56
Digger has a chance of selling water.
City: A  Money: $99   Fullness: 88  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier X for 0.9875623439452149.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 85  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 89  Prices: {'water': '$1.0', 'fertilizer': '$1.0', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 9} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $99   Fullness: 77  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $101  Fullness: 83  Prices: {'water': '$0.98', 'fertilizer': '$0.94', 'apple': '$1.01'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 1} Action: Carrier Y sold fertilizer to Farmer Joe for 0.99500625.

Day 57
Digger has a chance of selling water.
City: A  Money: $99   Fullness: 87  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 84  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 88  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.04'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 8} Action: Farmer Joe sold apple to Carrier X for 1.0343481099896192.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $98   Fullness: 76  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 1} Action: Carrier X refuses to buy apple from Carrier Y because the price is too high.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $102  Fullness: 82  Prices: {'water': '$0.98', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y sold apple to Farmer Joe for 1.0369404611424755.

Day 58
Digger has a chance of selling water.
City: A  Money: $99   Fullness: 86  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 83  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 8} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $98   Fullness: 75  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 1} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: B  Money: $102  Fullness: 81  Prices: {'water': '$0.98', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to B.

Day 59
Digger has a chance of selling water.
City: A  Money: $99   Fullness: 85  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
City: B  Money: $99   Fullness: 82  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.08'} Inventory: {'water': 1, 'fertilizer': 4, 'apple': 0} Action: Dirt refuses to buy apple from Carrier X because the price is too high.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 7} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $99   Fullness: 74  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$1.08'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Carrier X sold apple to Dirt for 1.0752455199543698.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
City: C  Money: $102  Fullness: 80  Prices: {'water': '$0.98', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to C.

Day 60
Digger has a chance of selling water.
City: A  Money: $99   Fullness: 84  Prices: {'water': '$1.03', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 81  Prices: {'water': '$0.95', 'fertilizer': '$0.98', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 3, 'apple': 1} Action: Dirt sold fertilizer to Carrier X for 0.938184226747954.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$0.99'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 6} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling fertilizer.
City: C  Money: $98   Fullness: 73  Prices: {'water': '$0.94', 'fertilizer': '$0.89', 'apple': '$1.08'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
City: A  Money: $102  Fullness: 79  Prices: {'water': '$0.98', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to A.

Day 61
Digger has a chance of selling water.
City: A  Money: $99   Fullness: 83  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Digger refuses to sell water to Carrier Y because the price is too low.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 80  Prices: {'water': '$0.95', 'fertilizer': '$0.98', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 3, 'apple': 1} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 5} Action: Farmer Joe sold apple to Carrier X for 1.0833503517003773.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $97   Fullness: 92  Prices: {'water': '$0.94', 'fertilizer': '$0.89', 'apple': '$1.03'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $102  Fullness: 78  Prices: {'water': '$0.98', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to C.

Day 62
Digger has a chance of selling water.
City: A  Money: $99   Fullness: 82  Prices: {'water': '$0.98', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 79  Prices: {'water': '$0.95', 'fertilizer': '$0.98', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 3, 'apple': 1} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$0.95', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 4} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: C  Money: $98   Fullness: 91  Prices: {'water': '$0.99', 'fertilizer': '$0.89', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 2, 'apple': 0} Action: Carrier X sold water to Farmer Joe for 0.99500625.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: A  Money: $102  Fullness: 77  Prices: {'water': '$0.98', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to A.

Day 63
Digger has a chance of selling water.
City: A  Money: $100  Fullness: 81  Prices: {'water': '$1.02', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier Y for 0.980174127728913.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $99   Fullness: 98  Prices: {'water': '$0.95', 'fertilizer': '$0.98', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 3, 'apple': 0} Action: Dirt consumed apple.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 99  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 14} Action: Farmer Joe grew 10 units of apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $99   Fullness: 90  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X sold fertilizer to Farmer Joe for 0.9925187343749999.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: B  Money: $101  Fullness: 76  Prices: {'water': '$0.93', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to B.

Day 64
Digger has a chance of selling water.
City: A  Money: $100  Fullness: 80  Prices: {'water': '$1.02', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $99   Fullness: 97  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 3, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier Y because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 13} Action: Farmer Joe sold apple to Carrier X for 1.0291828341153584.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: C  Money: $98   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$0.98'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
City: B  Money: $100  Fullness: 75  Prices: {'water': '$0.93', 'fertilizer': '$0.9', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y bought fertilizer from Dirt for 0.9288375077891111.

Day 65
Digger has a chance of selling water.
City: A  Money: $100  Fullness: 79  Prices: {'water': '$1.02', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 96  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier Y because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $100  Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 12} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $99   Fullness: 99  Prices: {'water': '$0.99', 'fertilizer': '$0.98', 'apple': '$0.98'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X sold fertilizer to Farmer Joe for 0.9428927976562499.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
City: A  Money: $100  Fullness: 74  Prices: {'water': '$0.93', 'fertilizer': '$0.9', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to A.

Day 66
Digger has a chance of selling water.
City: A  Money: $100  Fullness: 78  Prices: {'water': '$0.97', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Digger refuses to sell water to Carrier Y because the price is too low.
Dirt has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 95  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 99  Prices: {'water': '$0.99', 'fertilizer': '$0.9', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 12} Action: Farmer Joe refuses to sell apple to Carrier X because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
City: A  Money: $99   Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$0.98', 'apple': '$0.98'} Inventory: {'water': 2, 'fertilizer': 0, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $101  Fullness: 73  Prices: {'water': '$0.93', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Carrier Y sold fertilizer to Carrier X for 0.9826307044901383.

Day 67
Digger has a chance of selling water.
City: A  Money: $101  Fullness: 77  Prices: {'water': '$1.02', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier X for 0.9850934380853519.
Dirt has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 94  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$0.9', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 12} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $96   Fullness: 97  Prices: {'water': '$0.89', 'fertilizer': '$0.93', 'apple': '$0.98'} Inventory: {'water': 4, 'fertilizer': 1, 'apple': 0} Action: Carrier X bought water from Carrier Y for 0.9311654213424673.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $103  Fullness: 72  Prices: {'water': '$1.03', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier Y sold water to Digger for 1.021483243956651.

Day 68
Digger has a chance of selling water.
City: A  Money: $101  Fullness: 76  Prices: {'water': '$1.02', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier Y for 1.0266098770300702.
Dirt has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 93  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$0.9', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 12} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $97   Fullness: 96  Prices: {'water': '$0.93', 'fertilizer': '$0.93', 'apple': '$0.98'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Carrier X sold water to Digger for 1.0189295358467594.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $101  Fullness: 71  Prices: {'water': '$0.93', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Carrier Y bought water from Carrier X for 0.9334991692656315.

Day 69
Digger has a chance of selling water.
City: A  Money: $101  Fullness: 75  Prices: {'water': '$1.02', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier X for 0.9801741277289131.
Dirt has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 92  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 96  Prices: {'water': '$0.99', 'fertilizer': '$0.9', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 12} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $96   Fullness: 95  Prices: {'water': '$0.88', 'fertilizer': '$0.93', 'apple': '$0.98'} Inventory: {'water': 4, 'fertilizer': 1, 'apple': 0} Action: Carrier X bought water from Carrier Y for 0.9265154140196383.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $102  Fullness: 70  Prices: {'water': '$0.97', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to C.

Day 70
City: A  Money: $101  Fullness: 74  Prices: {'water': '$0.97', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 10, 'fertilizer': 1, 'apple': 0} Action: Digger collected 10 units of water.
Dirt has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 91  Prices: {'water': '$0.95', 'fertilizer': '$0.93', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.9', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 11} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $96   Fullness: 94  Prices: {'water': '$0.88', 'fertilizer': '$0.93', 'apple': '$0.98'} Inventory: {'water': 4, 'fertilizer': 1, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
City: C  Money: $103  Fullness: 69  Prices: {'water': '$1.02', 'fertilizer': '$0.94', 'apple': '$1.06'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier Y sold water to Farmer Joe for 0.9925187343749999.

Day 71
Digger has a chance of selling water.
City: A  Money: $101  Fullness: 73  Prices: {'water': '$0.97', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 10, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $101  Fullness: 90  Prices: {'water': '$0.95', 'fertilizer': '$0.97', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Dirt sold fertilizer to Carrier X for 0.9334991692656314.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $98   Fullness: 100 Prices: {'water': '$0.94', 'fertilizer': '$0.9', 'apple': '$0.98'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 10} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $95   Fullness: 93  Prices: {'water': '$0.93', 'fertilizer': '$0.89', 'apple': '$0.98'} Inventory: {'water': 4, 'fertilizer': 2, 'apple': 0} Action: Carrier X refuses to buy water from Dirt because the price is too high.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
City: C  Money: $102  Fullness: 68  Prices: {'water': '$1.02', 'fertilizer': '$0.89', 'apple': '$1.06'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Carrier Y bought fertilizer from Farmer Joe for 0.8957481577734373.

Day 72
Digger has a chance of selling water.
City: A  Money: $101  Fullness: 72  Prices: {'water': '$0.97', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 10, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $101  Fullness: 89  Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $99   Fullness: 99  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$0.93'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 20} Action: Farmer Joe grew 10 units of apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: B  Money: $95   Fullness: 92  Prices: {'water': '$0.93', 'fertilizer': '$0.93', 'apple': '$0.98'} Inventory: {'water': 4, 'fertilizer': 2, 'apple': 0} Action: Carrier X refuses to buy fertilizer from Dirt because the price is too high.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $101  Fullness: 67  Prices: {'water': '$1.02', 'fertilizer': '$0.89', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 1} Action: Carrier Y bought apple from Farmer Joe for 0.931165421342467.

Day 73
Digger has a chance of selling water.
City: A  Money: $101  Fullness: 71  Prices: {'water': '$0.97', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 10, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $102  Fullness: 88  Prices: {'water': '$0.95', 'fertilizer': '$0.97', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 0} Action: Dirt sold fertilizer to Carrier X for 0.9311654213424673.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$1.03'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 18} Action: Farmer Joe sold apple to Carrier Y for 1.0037407996339962.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $94   Fullness: 91  Prices: {'water': '$0.98', 'fertilizer': '$0.88', 'apple': '$0.98'} Inventory: {'water': 4, 'fertilizer': 3, 'apple': 0} Action: Carrier X refuses to buy water from Dirt because the price is too high.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $100  Fullness: 86  Prices: {'water': '$1.02', 'fertilizer': '$0.89', 'apple': '$0.95'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 1} Action: Carrier Y consumed apple.

Day 74
Digger has a chance of selling water.
City: A  Money: $101  Fullness: 70  Prices: {'water': '$0.97', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 10, 'fertilizer': 1, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
City: B  Money: $102  Fullness: 87  Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$1.02'} Inventory: {'water': 1, 'fertilizer': 10, 'apple': 0} Action: Dirt produced 10 units of fertilizer.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 18} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: A  Money: $94   Fullness: 90  Prices: {'water': '$0.98', 'fertilizer': '$0.88', 'apple': '$0.98'} Inventory: {'water': 4, 'fertilizer': 3, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of consuming an apple.
City: B  Money: $100  Fullness: 85  Prices: {'water': '$1.02', 'fertilizer': '$0.89', 'apple': '$0.95'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 1} Action: Carrier Y moved to B.

Day 75
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 69  Prices: {'water': '$1.01', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 9, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier X for 0.9752793831785669.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
City: B  Money: $101  Fullness: 86  Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$0.97'} Inventory: {'water': 1, 'fertilizer': 10, 'apple': 1} Action: Dirt bought apple from Carrier Y for 0.9535537596522964.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 96  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$0.98'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 18} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $93   Fullness: 89  Prices: {'water': '$0.93', 'fertilizer': '$0.88', 'apple': '$0.98'} Inventory: {'water': 5, 'fertilizer': 3, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
City: A  Money: $101  Fullness: 84  Prices: {'water': '$1.02', 'fertilizer': '$0.89', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to A.

Day 76
Digger has a chance of selling water.
City: A  Money: $103  Fullness: 68  Prices: {'water': '$1.06', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier Y for 1.0214832439566515.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $101  Fullness: 85  Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$0.97'} Inventory: {'water': 1, 'fertilizer': 10, 'apple': 1} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 95  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 17} Action: Farmer Joe sold apple to Carrier X for 0.9777236924095905.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $93   Fullness: 88  Prices: {'water': '$0.93', 'fertilizer': '$0.88', 'apple': '$0.98'} Inventory: {'water': 5, 'fertilizer': 3, 'apple': 0} Action: Carrier X sold apple to Farmer Joe for 1.0240433523374948.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $101  Fullness: 83  Prices: {'water': '$1.02', 'fertilizer': '$0.89', 'apple': '$1.0'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Carrier Y sold water to Digger for 1.064533319300981.

Day 77
Digger has a chance of selling water.
City: A  Money: $103  Fullness: 67  Prices: {'water': '$1.06', 'fertilizer': '$0.95', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 1, 'apple': 0} Action: Digger sold water to Carrier Y for 1.0189295358467598.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $101  Fullness: 84  Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$0.97'} Inventory: {'water': 1, 'fertilizer': 10, 'apple': 1} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 94  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 17} Action: Farmer Joe sold apple to Carrier X for 0.9752793831785664.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: C  Money: $92   Fullness: 100 Prices: {'water': '$0.93', 'fertilizer': '$0.88', 'apple': '$0.93'} Inventory: {'water': 5, 'fertilizer': 3, 'apple': 0} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $101  Fullness: 82  Prices: {'water': '$0.97', 'fertilizer': '$0.94', 'apple': '$1.0'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y sold fertilizer to Digger for 0.9476249999999999.

Day 78
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 66  Prices: {'water': '$1.01', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger refuses to sell water to Carrier Y because the price is too low.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $101  Fullness: 83  Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$0.97'} Inventory: {'water': 1, 'fertilizer': 10, 'apple': 1} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 93  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$0.97'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 17} Action: Farmer Joe refuses to sell apple to Carrier X because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: A  Money: $92   Fullness: 99  Prices: {'water': '$0.93', 'fertilizer': '$0.88', 'apple': '$0.93'} Inventory: {'water': 5, 'fertilizer': 3, 'apple': 0} Action: Carrier X moved to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: C  Money: $101  Fullness: 81  Prices: {'water': '$0.97', 'fertilizer': '$0.94', 'apple': '$1.0'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to C.

Day 79
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 65  Prices: {'water': '$0.96', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger refuses to sell water to Carrier X because the price is too low.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $101  Fullness: 100 Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$0.97'} Inventory: {'water': 1, 'fertilizer': 10, 'apple': 0} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 92  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 16} Action: Farmer Joe sold apple to Carrier Y for 1.0012314476349113.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: B  Money: $92   Fullness: 98  Prices: {'water': '$0.93', 'fertilizer': '$0.88', 'apple': '$0.93'} Inventory: {'water': 5, 'fertilizer': 3, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: B  Money: $100  Fullness: 80  Prices: {'water': '$0.97', 'fertilizer': '$0.94', 'apple': '$0.95'} Inventory: {'water': 2, 'fertilizer': 1, 'apple': 1} Action: Carrier Y moved to B.

Day 80
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 64  Prices: {'water': '$0.96', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
City: B  Money: $100  Fullness: 99  Prices: {'water': '$0.95', 'fertilizer': '$0.92', 'apple': '$0.92'} Inventory: {'water': 1, 'fertilizer': 10, 'apple': 1} Action: Dirt bought apple from Carrier Y for 0.9511698752531658.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 91  Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 16} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
City: C  Money: $92   Fullness: 97  Prices: {'water': '$0.93', 'fertilizer': '$0.88', 'apple': '$0.93'} Inventory: {'water': 5, 'fertilizer': 3, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
City: B  Money: $100  Fullness: 79  Prices: {'water': '$0.92', 'fertilizer': '$0.94', 'apple': '$1.0'} Inventory: {'water': 3, 'fertilizer': 1, 'apple': 0} Action: Carrier Y bought water from Dirt for 0.9476249999999999.

Day 81
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 63  Prices: {'water': '$0.96', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $102  Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$0.97', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 1} Action: Dirt sold fertilizer to Carrier Y for 0.9381842267479539.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.99', 'apple': '$1.02'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 15} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $93   Fullness: 96  Prices: {'water': '$0.93', 'fertilizer': '$0.93', 'apple': '$0.93'} Inventory: {'water': 5, 'fertilizer': 2, 'apple': 0} Action: Carrier X sold fertilizer to Farmer Joe for 0.9875623439452147.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $99   Fullness: 78  Prices: {'water': '$0.92', 'fertilizer': '$0.89', 'apple': '$1.0'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to C.

Day 82
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 62  Prices: {'water': '$0.96', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of consuming an apple.
City: B  Money: $102  Fullness: 100 Prices: {'water': '$1.0', 'fertilizer': '$0.97', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 0} Action: Dirt consumed apple.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 99  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$0.97'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 15} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: C  Money: $94   Fullness: 95  Prices: {'water': '$0.93', 'fertilizer': '$0.98', 'apple': '$0.93'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 0} Action: Carrier X sold fertilizer to Farmer Joe for 0.9381842267479539.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $99   Fullness: 77  Prices: {'water': '$0.92', 'fertilizer': '$0.89', 'apple': '$1.0'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to A.

Day 83
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 61  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger refuses to sell water to Carrier Y because the price is too low.
Dirt has a chance of selling fertilizer.
City: B  Money: $102  Fullness: 99  Prices: {'water': '$1.0', 'fertilizer': '$0.97', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $101  Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$0.89', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 15} Action: Farmer Joe refuses to sell apple to Carrier X because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
City: C  Money: $93   Fullness: 94  Prices: {'water': '$0.93', 'fertilizer': '$0.98', 'apple': '$0.88'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 1} Action: Carrier X bought apple from Farmer Joe for 0.9195839061017003.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
City: C  Money: $99   Fullness: 76  Prices: {'water': '$0.92', 'fertilizer': '$0.89', 'apple': '$1.0'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to C.

Day 84
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 60  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $102  Fullness: 98  Prices: {'water': '$1.0', 'fertilizer': '$0.97', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$0.89', 'apple': '$1.01'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 13} Action: Farmer Joe sold apple to Carrier Y for 0.9987283690158241.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling apple.
Carrier X has a chance of consuming an apple.
City: B  Money: $93   Fullness: 93  Prices: {'water': '$0.93', 'fertilizer': '$0.98', 'apple': '$0.88'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 1} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $98   Fullness: 95  Prices: {'water': '$0.92', 'fertilizer': '$0.89', 'apple': '$0.95'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y consumed apple.

Day 85
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 59  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
City: B  Money: $101  Fullness: 97  Prices: {'water': '$1.0', 'fertilizer': '$0.97', 'apple': '$0.88'} Inventory: {'water': 0, 'fertilizer': 9, 'apple': 1} Action: Dirt bought apple from Carrier X for 0.8801896433186561.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 96  Prices: {'water': '$0.99', 'fertilizer': '$0.89', 'apple': '$0.96'} Inventory: {'water': 0, 'fertilizer': 2, 'apple': 13} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $93   Fullness: 92  Prices: {'water': '$0.93', 'fertilizer': '$0.98', 'apple': '$0.88'} Inventory: {'water': 5, 'fertilizer': 1, 'apple': 1} Action: Carrier X bought apple from Dirt for 0.8757941962873338.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: C  Money: $97   Fullness: 94  Prices: {'water': '$0.92', 'fertilizer': '$0.85', 'apple': '$0.95'} Inventory: {'water': 3, 'fertilizer': 3, 'apple': 0} Action: Carrier Y bought fertilizer from Farmer Joe for 0.8912750154105561.

Day 86
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 58  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
Dirt has a chance of buying an apple.
City: B  Money: $103  Fullness: 96  Prices: {'water': '$1.0', 'fertilizer': '$1.02', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 0} Action: Dirt sold fertilizer to Carrier X for 0.9752793831785667.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $104  Fullness: 95  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$0.91'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 13} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: B  Money: $92   Fullness: 100 Prices: {'water': '$0.93', 'fertilizer': '$0.93', 'apple': '$0.88'} Inventory: {'water': 5, 'fertilizer': 2, 'apple': 0} Action: Carrier X consumed apple.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
City: B  Money: $97   Fullness: 93  Prices: {'water': '$0.92', 'fertilizer': '$0.85', 'apple': '$0.95'} Inventory: {'water': 3, 'fertilizer': 3, 'apple': 0} Action: Carrier Y moved to B.

Day 87
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 57  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 95  Prices: {'water': '$1.0', 'fertilizer': '$0.97', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $104  Fullness: 94  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$0.91'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 13} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $91   Fullness: 99  Prices: {'water': '$0.93', 'fertilizer': '$0.88', 'apple': '$0.88'} Inventory: {'water': 5, 'fertilizer': 3, 'apple': 0} Action: Carrier X bought fertilizer from Carrier Y for 0.8467112646400282.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $98   Fullness: 92  Prices: {'water': '$0.92', 'fertilizer': '$0.89', 'apple': '$0.95'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y moved to C.

Day 88
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 56  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 94  Prices: {'water': '$1.0', 'fertilizer': '$0.92', 'apple': '$0.92'} Inventory: {'water': 0, 'fertilizer': 8, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $105  Fullness: 93  Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$0.96'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 12} Action: Farmer Joe sold apple to Carrier Y for 0.9487919505650328.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $92   Fullness: 98  Prices: {'water': '$0.97', 'fertilizer': '$0.88', 'apple': '$0.88'} Inventory: {'water': 4, 'fertilizer': 3, 'apple': 0} Action: Carrier X sold water to Dirt for 0.99500625.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of selling apple.
Carrier Y has a chance of consuming an apple.
City: C  Money: $97   Fullness: 100 Prices: {'water': '$0.92', 'fertilizer': '$0.89', 'apple': '$0.9'} Inventory: {'water': 3, 'fertilizer': 2, 'apple': 0} Action: Carrier Y consumed apple.

Day 89
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 55  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $102  Fullness: 93  Prices: {'water': '$0.95', 'fertilizer': '$0.87', 'apple': '$0.92'} Inventory: {'water': 1, 'fertilizer': 8, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $105  Fullness: 100 Prices: {'water': '$0.99', 'fertilizer': '$0.94', 'apple': '$0.96'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 11} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: B  Money: $91   Fullness: 97  Prices: {'water': '$0.97', 'fertilizer': '$0.84', 'apple': '$0.88'} Inventory: {'water': 4, 'fertilizer': 4, 'apple': 0} Action: Carrier X bought fertilizer from Dirt for 0.871420699019624.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $98   Fullness: 99  Prices: {'water': '$0.97', 'fertilizer': '$0.89', 'apple': '$0.9'} Inventory: {'water': 2, 'fertilizer': 2, 'apple': 0} Action: Carrier Y sold water to Farmer Joe for 0.9900374375390625.

Day 90
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 54  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 92  Prices: {'water': '$0.95', 'fertilizer': '$0.87', 'apple': '$0.92'} Inventory: {'water': 1, 'fertilizer': 7, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $104  Fullness: 99  Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$0.91'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 11} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: B  Money: $91   Fullness: 96  Prices: {'water': '$0.92', 'fertilizer': '$0.84', 'apple': '$0.88'} Inventory: {'water': 4, 'fertilizer': 4, 'apple': 0} Action: Carrier X refuses to sell water to Dirt because the price is too low.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $98   Fullness: 98  Prices: {'water': '$0.92', 'fertilizer': '$0.89', 'apple': '$0.9'} Inventory: {'water': 2, 'fertilizer': 2, 'apple': 0} Action: Carrier Y refuses to sell water to Farmer Joe because the price is too low.

Day 91
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 53  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 91  Prices: {'water': '$0.95', 'fertilizer': '$0.83', 'apple': '$0.92'} Inventory: {'water': 1, 'fertilizer': 7, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $104  Fullness: 100 Prices: {'water': '$0.94', 'fertilizer': '$0.94', 'apple': '$0.91'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 10} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: B  Money: $91   Fullness: 95  Prices: {'water': '$0.92', 'fertilizer': '$0.79', 'apple': '$0.88'} Inventory: {'water': 4, 'fertilizer': 5, 'apple': 0} Action: Carrier X bought fertilizer from Dirt for 0.8257800399084712.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $99   Fullness: 97  Prices: {'water': '$0.96', 'fertilizer': '$0.89', 'apple': '$0.9'} Inventory: {'water': 1, 'fertilizer': 2, 'apple': 0} Action: Carrier Y sold water to Farmer Joe for 0.9405355656621093.

Day 92
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 52  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 90  Prices: {'water': '$0.95', 'fertilizer': '$0.82', 'apple': '$0.92'} Inventory: {'water': 1, 'fertilizer': 6, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 99  Prices: {'water': '$0.94', 'fertilizer': '$0.98', 'apple': '$0.87'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 20} Action: Farmer Joe grew 10 units of apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: B  Money: $91   Fullness: 94  Prices: {'water': '$0.92', 'fertilizer': '$0.83', 'apple': '$0.88'} Inventory: {'water': 4, 'fertilizer': 5, 'apple': 0} Action: Carrier X refuses to buy fertilizer from Dirt because the price is too high.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $100  Fullness: 96  Prices: {'water': '$0.96', 'fertilizer': '$0.93', 'apple': '$0.9'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier Y sold fertilizer to Farmer Joe for 0.9826307044901382.

Day 93
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 51  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $104  Fullness: 89  Prices: {'water': '$0.95', 'fertilizer': '$0.86', 'apple': '$0.92'} Inventory: {'water': 1, 'fertilizer': 5, 'apple': 0} Action: Dirt sold fertilizer to Carrier X for 0.8340897107498417.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 100 Prices: {'water': '$0.94', 'fertilizer': '$0.93', 'apple': '$0.87'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 19} Action: Farmer Joe consumed apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying fertilizer.
Carrier X has a chance of selling water.
City: B  Money: $91   Fullness: 93  Prices: {'water': '$0.97', 'fertilizer': '$0.79', 'apple': '$0.88'} Inventory: {'water': 3, 'fertilizer': 6, 'apple': 0} Action: Carrier X sold water to Dirt for 0.9452559374999999.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: C  Money: $99   Fullness: 95  Prices: {'water': '$0.96', 'fertilizer': '$0.93', 'apple': '$0.86'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 1} Action: Carrier Y bought apple from Farmer Joe for 0.8670690419038943.

Day 94
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 50  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 88  Prices: {'water': '$0.9', 'fertilizer': '$0.82', 'apple': '$0.92'} Inventory: {'water': 2, 'fertilizer': 5, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 99  Prices: {'water': '$0.94', 'fertilizer': '$0.93', 'apple': '$0.86'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 18} Action: Farmer Joe refuses to sell apple to Carrier Y because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: C  Money: $91   Fullness: 92  Prices: {'water': '$0.97', 'fertilizer': '$0.79', 'apple': '$0.88'} Inventory: {'water': 3, 'fertilizer': 6, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of consuming an apple.
City: C  Money: $100  Fullness: 94  Prices: {'water': '$1.01', 'fertilizer': '$0.93', 'apple': '$0.86'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 1} Action: Carrier Y sold water to Carrier X for 0.970409081758819.

Day 95
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 49  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$1.02'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 0} Action: Digger tried to sell water, but there are no buyers in A.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 87  Prices: {'water': '$0.9', 'fertilizer': '$0.82', 'apple': '$0.92'} Inventory: {'water': 2, 'fertilizer': 5, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of growing apples.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 98  Prices: {'water': '$0.99', 'fertilizer': '$0.98', 'apple': '$0.82'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 28} Action: Farmer Joe grew 10 units of apple.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $90   Fullness: 91  Prices: {'water': '$0.92', 'fertilizer': '$0.79', 'apple': '$0.88'} Inventory: {'water': 4, 'fertilizer': 6, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling fertilizer.
Carrier Y has a chance of consuming an apple.
City: A  Money: $100  Fullness: 93  Prices: {'water': '$1.01', 'fertilizer': '$0.93', 'apple': '$0.86'} Inventory: {'water': 0, 'fertilizer': 1, 'apple': 1} Action: Carrier Y moved to A.

Day 96
Digger has a chance of selling water.
Digger has a chance of buying an apple.
City: A  Money: $101  Fullness: 48  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$0.97'} Inventory: {'water': 8, 'fertilizer': 2, 'apple': 1} Action: Digger bought apple from Carrier Y for 0.8562847353849421.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 86  Prices: {'water': '$0.9', 'fertilizer': '$0.78', 'apple': '$0.92'} Inventory: {'water': 2, 'fertilizer': 5, 'apple': 0} Action: Dirt refuses to sell fertilizer to Carrier X because the price is too low.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 97  Prices: {'water': '$0.99', 'fertilizer': '$0.98', 'apple': '$0.82'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 28} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of buying fertilizer.
City: C  Money: $90   Fullness: 90  Prices: {'water': '$0.92', 'fertilizer': '$0.79', 'apple': '$0.88'} Inventory: {'water': 4, 'fertilizer': 6, 'apple': 0} Action: Carrier X moved to C.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $100  Fullness: 92  Prices: {'water': '$0.96', 'fertilizer': '$0.93', 'apple': '$0.9'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier Y bought water from Digger for 0.9104224939990893.

Day 97
Digger has a chance of selling water.
Digger has a chance of consuming an apple.
City: A  Money: $102  Fullness: 67  Prices: {'water': '$0.96', 'fertilizer': '$0.9', 'apple': '$0.97'} Inventory: {'water': 7, 'fertilizer': 2, 'apple': 0} Action: Digger consumed apple.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 85  Prices: {'water': '$0.9', 'fertilizer': '$0.78', 'apple': '$0.92'} Inventory: {'water': 2, 'fertilizer': 5, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $103  Fullness: 96  Prices: {'water': '$0.99', 'fertilizer': '$0.98', 'apple': '$0.86'} Inventory: {'water': 0, 'fertilizer': 0, 'apple': 27} Action: Farmer Joe sold apple to Carrier X for 0.8779891692103594.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: C  Money: $90   Fullness: 89  Prices: {'water': '$0.97', 'fertilizer': '$0.79', 'apple': '$0.83'} Inventory: {'water': 3, 'fertilizer': 6, 'apple': 1} Action: Carrier X sold water to Farmer Joe for 0.9850934380853518.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: A  Money: $100  Fullness: 91  Prices: {'water': '$0.91', 'fertilizer': '$0.93', 'apple': '$0.9'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier Y refuses to sell water to Digger because the price is too low.

Day 98
Digger has a chance of selling water.
City: A  Money: $102  Fullness: 66  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$0.97'} Inventory: {'water': 7, 'fertilizer': 2, 'apple': 0} Action: Digger refuses to sell water to Carrier Y because the price is too low.
Dirt has a chance of selling fertilizer.
City: B  Money: $103  Fullness: 84  Prices: {'water': '$0.9', 'fertilizer': '$0.78', 'apple': '$0.92'} Inventory: {'water': 2, 'fertilizer': 5, 'apple': 0} Action: Dirt tried to sell fertilizer, but there are no buyers in B.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 95  Prices: {'water': '$0.94', 'fertilizer': '$0.98', 'apple': '$0.82'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 27} Action: Farmer Joe refuses to sell apple to Carrier X because the price is too low.
Carrier X has a chance of moving to A.
Carrier X has a chance of moving to B.
Carrier X has a chance of buying apple.
Carrier X has a chance of selling water.
Carrier X has a chance of selling fertilizer.
Carrier X has a chance of consuming an apple.
City: A  Money: $90   Fullness: 88  Prices: {'water': '$0.97', 'fertilizer': '$0.79', 'apple': '$0.83'} Inventory: {'water': 3, 'fertilizer': 6, 'apple': 1} Action: Carrier X moved to A.
Carrier Y has a chance of moving to B.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of buying apple.
Carrier Y has a chance of selling water.
Carrier Y has a chance of selling fertilizer.
City: B  Money: $100  Fullness: 90  Prices: {'water': '$0.91', 'fertilizer': '$0.93', 'apple': '$0.9'} Inventory: {'water': 1, 'fertilizer': 1, 'apple': 0} Action: Carrier Y moved to B.

Day 99
Digger has a chance of selling water.
Digger has a chance of buying an apple.
City: A  Money: $101  Fullness: 65  Prices: {'water': '$0.91', 'fertilizer': '$0.9', 'apple': '$0.92'} Inventory: {'water': 7, 'fertilizer': 2, 'apple': 1} Action: Digger bought apple from Carrier X for 0.8340897107498414.
Dirt has a chance of selling fertilizer.
City: B  Money: $104  Fullness: 83  Prices: {'water': '$0.9', 'fertilizer': '$0.82', 'apple': '$0.92'} Inventory: {'water': 2, 'fertilizer': 4, 'apple': 0} Action: Dirt sold fertilizer to Carrier Y for 0.9334991692656313.
Farmer Joe has a chance of selling apples.
Farmer Joe has a chance of consuming an apple.
City: C  Money: $102  Fullness: 94  Prices: {'water': '$0.94', 'fertilizer': '$0.98', 'apple': '$0.82'} Inventory: {'water': 1, 'fertilizer': 0, 'apple': 27} Action: Farmer Joe tried to sell apple, but there are no buyers in C.
Carrier X has a chance of moving to B.
Carrier X has a chance of moving to C.
Carrier X has a chance of buying water.
Carrier X has a chance of selling fertilizer.
City: B  Money: $91   Fullness: 87  Prices: {'water': '$0.97', 'fertilizer': '$0.79', 'apple': '$0.88'} Inventory: {'water': 3, 'fertilizer': 6, 'apple': 0} Action: Carrier X moved to B.
Carrier Y has a chance of moving to A.
Carrier Y has a chance of moving to C.
Carrier Y has a chance of buying water.
Carrier Y has a chance of buying fertilizer.
Carrier Y has a chance of selling water.
City: B  Money: $98   Fullness: 89  Prices: {'water': '$0.91', 'fertilizer': '$0.84', 'apple': '$0.9'} Inventory: {'water': 1, 'fertilizer': 3, 'apple': 0} Action: Carrier Y bought fertilizer from Carrier X for 0.7923852252123496.

