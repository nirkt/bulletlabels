# Label rules

What counts as a label, and how it is shaped. Every rule here was decided by a person looking at the real frames; the labeller
(`labeller/idlabels.py` + `labeller/fixes/*.py`) implements them, and 740 review pages (every brawler and skin, solo mode,
attacks at 0 and 90 degrees and the super) were checked against them. The same data is in `data/label_decisions.json`.

## Classes

| code | class | outline colour (overlays / review pages) |
|---|---|---|
| 1 | attack | red |
| 2 | super | magenta |
| 3 | spawnable (a character a brawler spawns: turret, pet, ...) | blue |
| 4 | zone (a lingering damage / effect area) | yellow |
| 5 | gadget / star power | cyan |
| 6 | spawnable attack | orange |
| 7 | target (landing marker) | white |
| 8 | throwable (no damage) | pink |
| 9 | pull | violet |

Every instance also carries its team (own = the camera's brawler, enemy = everyone else), its type name, and the brawler and skin
that fired it, so other class schemes can be made later (`python3 bl.py classes`). Enemy shots are drawn red-tinted by the game;
their labels follow the same rules as the blue (own) ones and were checked to be identical shape for shape.

## General rules

* Label = the object's own outline, filled solid: no holes, spurs, specks; no glow / halo; nothing cut off.
* Hidden parts are filled in (smoke / dust / flash covering a projectile): the label keeps the full shape.
* Never labelled: explosions, hit flashes, muzzle flashes, end / fade animations, leftovers after the attack's logic ended, brawlers, UI, walls, bushes.
* Thin lines / slashes / streaks = ONE convex hull over the whole attack.
* Zones (yellow) = the full logic circle from zone birth, cut only by walls, crates, barrels.
* Separate projectiles = separate labels (except listed one-label-while-touching cases).

## Later decisions (applied everywhere)

* Glow / halo layers are not labelled (no separate glow class).
* Attacks that have ended keep their label only while their own pieces are still visible; explosions and hit flashes never.
* A small piece far from the rest of the attack gets no label (implemented for the cases found: Ash's attacks; Ruffs 427 bone chain).
* Attacks drawn as a hull keep their hull.
* One attack = one label: a piece that the labeller had handed to an earlier, already ended shot of the same brawler goes to the shot
  it belongs to.
* Labels are at 1200 x 540 (half the game's 2400 x 1080); a pixel belongs to a label when the object covers at least half of it.

## Brawler-wide decisions

(internal names: Knight = Ash, Gunslinger = Colt, ShotgunGirl = Shelly, ... - `python3 bl.py skins`)

* **ArtilleryDude**: Penny super: cannon blue, bomb orange, landing ring white (drawn size), burn zone yellow only after the bomb lands.
* **ArtilleryGal**: Penny super: cannon blue, bomb orange, landing ring white (drawn size), burn zone yellow only after the bomb lands.
* **BanditGirl**: Shelly: NO hull for her bullets; each bullet in its own filled outline.
* **Barkeep**: Barley: puddle = the full logic circle minus walls / crates.
* **Blackhole**: Tara super: eyeball = pink throwable, pull vortex = violet pull, final explosion = yellow zone circle.
* **Blower**: Gale: attack = snowballs only (heads, no curls).
* **BowDude**: Bo: Wasp skin = the whole arrow as one label.
* **Cactus**: Spike: Mecha skin = ball only, no chain.
* **DeadMariachi**: Poco: one solid zone; the label stops when the logic / hitbox ends (no leftovers).
* **Frank**: Frank: v7 template per skin (one hull over the wave rows still active; old rows drop out; nothing after the wave ends; bushes cut it, characters don't).
* **GunSlinger**: Colt super: each bullet separately, core of the comet head only (no outer glow).
* **Gunslinger**: Colt super: each bullet separately, core of the comet head only (no outer glow).
* **HammerDude**: Frank: v7 template per skin (one hull over the wave rows still active; old rows drop out; nothing after the wave ends; bushes cut it, characters don't).
* **JetpackGirl**: Janet: the super has NO label at all.
* **Meeple**: Zone keeps the dark circle drawn around the hexagon.
* **Mummy**: Emz: ONE label per attack (the 5 clouds of one shot), cloud-shaped filled outline.
* **Ninja**: Leon: bright blade edges are part of the blade; label = filled outline, no holes.
* **Poco**: Poco: one solid zone; the label stops when the logic / hitbox ends (no leftovers).
* **PowerLeveler**: Surge: the whole cyan blob incl. touching splash; detached dots out.
* **Sandstorm**: Sandy: one solid zone over the whole cloud; max-range burst unlabelled; Lantern posts filled.
* **ShotgunGirl**: Shelly: NO hull for her bullets; each bullet in its own filled outline.
* **Twins**: Twins super: star only, smooth edges (Angel: hull of the star).
* **Undertaker**: Mortis super: one hull.
* **Voodoo**: Landing zone: the dome as a disc while visible, then only the puddle.

## Per-skin decisions

* **005 DuplicatorDarkcircus** (Duplicator): The attack's tail is labelled too.
* **016 GunSlingerDefault** (Gunslinger): Colt super: core only, no bit on the front from the glow.
* **020 CactusDefault** (Cactus): Spike: the tiny projectiles flying out after the ball pops are labelled.
* **026 UndertakerDefault** (Undertaker): Mortis super: one hull.
* **028 DeadMariachiDefault** (DeadMariachi): Poco default: drop the clouds when the logic ends.
* **032 BlackholeDefault** (BlackHole): Tara super: pull stage = pull only, no zone yet.
* **041 ArcadeDefault** (Arcade): Arcade attack: one hull per projectile.
* **045 SpawnerDudeDefault** (SpawnerDude): Circle super: only the fully visible circle.
* **111 MenderDefault** (Mender): Mender: the whole shining circle.
* **113 RedirecterDefault** (Redirecter): Bullet tumbling into the water after it ends = end effect, unlabelled.
* **121 VoodooSandsOfTimes** (Voodoo): Attack: the thing in the middle is not labelled.
* **133 WeaponThrowerYeti** (WeaponThrower): Bronson Yeti: claw swoosh hull incl. front flash; claw marks after the swipe unlabelled.
* **184 RocketGirlRanger_red** (RocketGirl): Zone: full circle by the circle rule, ignore the brawler in it; tight to walls (no gap).
* **220 FrankCaveman** (HammerDude): Caveman super flame stub: not included.
* **242 HammerDudeDJ** (HammerDude): Frank DJ: keep the first blue row; tiles are a fixed pattern.
* **254 BlowerTrader** (Blower): Heads only, no curls; heads hidden by dust are filled in (separate labels unless they overlap a lot).
* **332 DuelistBrain** (Duelist): Cordelius Brain: neuron only.
* **380 TntDudeZombie** (TntDude): Touching labels drawn as one outline: leave as is.
* **382 ElectroSniperSCmake** (ElectroSniper): Belle: flame only, cyan sparks around it out.
* **400 ShieldTankSupercellMake** (ShieldTank): Atoms: ONE label while they touch.
* **402 SkaterSandsOfTimes** (Skater): Skater balls: one label while overlapping, separate once apart.
* **416 FireDudeFantasy** (FireDude): Amber: crystal base in.
* **424 SnakeOilSurfer** (SnakeOil): Byron: the pink flame tip may stay.
* **431 PocoSurferOvercharge** (DeadMariachi): Poco Surfer: discs only; one hull per shot; cut by walls they are drawn behind.
* **453 JetpackGirlMHA** (JetpackGirl): Janet crystal next to Janet is labelled (it ended at tick 307: say if it should go).
* **495 EnragerDuel** (Enrager): Enrager Duel: one label while the two projectiles overlap; hull of the bright band.
* **521 MechaDudeMonsterTruck** (MechaDude): Fists filled in under the puffs.
* **533 LuchadorButler** (Luchador): Hidden part filled in.
* **536 MummyHarajuku** (Mummy): Emz Mummy: one hull per stack of overlapping clouds.
* **585 CannonGirlChristmas** (CannonGirl): Cannon Girl: blue ribbon out.
* **588 RollerGoodRandoms** (Roller): Ice streak: separate label per projectile.
* **610 WhirlwindCarreta** (Whirlwind): Carl: crescent incl. teal end caps; NOT the green swoosh or glow.
* **612 CookerPorcelain** (Cooker): Cooker: pearl incl. leading white lobe and white bulbs.

## Not reviewed yet

* Ring mode (several brawlers in one capture) - see README.
* Gadgets and star powers in battle (the offline game cannot trigger gadgets).
