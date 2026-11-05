#include <catch2/catch_test_macros.hpp>
#include "pommerman/characters.hpp"

using namespace pommerman;

TEST_CASE("Bomb ticks and explodes", "[bomb]") {
    Bomb b;
    b.life = 3;

    CHECK(!b.exploded());
    b.tick(); CHECK(!b.exploded());
    b.tick(); CHECK(!b.exploded());
    b.tick(); CHECK(b.exploded());
}

TEST_CASE("Bomb fire() sets life to zero", "[bomb]") {
    Bomb b;
    b.fire();
    CHECK(b.exploded());
}

TEST_CASE("Bomb raw blast positions: no walls", "[bomb]") {
    Bomb b;
    b.position      = {5, 5};
    b.blast_strength = 2;
    auto positions = b.raw_blast_positions();

    // Centre + 2 in each of 4 directions = 9
    CHECK(positions.size() == 9);
    bool has_centre = false;
    for (auto& p : positions) if (p.row == 5 && p.col == 5) has_centre = true;
    CHECK(has_centre);
}

TEST_CASE("Bomb raw blast positions: clipped at board edge", "[bomb]") {
    Bomb b;
    b.position      = {0, 0};
    b.blast_strength = 3;
    auto positions = b.raw_blast_positions();

    // From (0,0): up 0, left 0, right 3, down 3 = 1 + 3 + 3 = 7
    CHECK(positions.size() == 7);
}

TEST_CASE("Bomb moving direction", "[bomb]") {
    Bomb b;
    CHECK(!b.is_moving());
    b.moving_direction = Direction::Right;
    CHECK(b.is_moving());
    b.stop();
    CHECK(!b.is_moving());
}

TEST_CASE("Agent set_agent_id FFA", "[agent]") {
    Agent a;
    a.set_agent_id(0, GameType::FFA);
    CHECK(a.agent_id == 0);
    CHECK(a.teammate_id == -1);
    // enemies: 1, 2, 3
    std::array<int,3> expected{1, 2, 3};
    CHECK(a.enemy_ids == expected);
}

TEST_CASE("Agent set_agent_id Team", "[agent]") {
    Agent a;
    a.set_agent_id(0, GameType::Team);
    CHECK(a.teammate_id == 1);
}

TEST_CASE("Agent maybe_lay_bomb decrements ammo", "[agent]") {
    Agent a;
    a.ammo     = 1;
    a.position = {3, 3};
    a.blast_strength = 2;

    auto bomb = a.maybe_lay_bomb();
    REQUIRE(bomb.has_value());
    CHECK(a.ammo == 0);
    CHECK(bomb->position.row == 3);
    CHECK(bomb->blast_strength == 2);

    auto no_bomb = a.maybe_lay_bomb();
    CHECK(!no_bomb.has_value());
}

TEST_CASE("Agent pick_up powerups", "[agent]") {
    Agent a;
    int initial_ammo  = a.ammo;
    int initial_range = a.blast_strength;

    a.pick_up(Item::ExtraBomb);
    CHECK(a.ammo == initial_ammo + 1);

    a.pick_up(Item::IncrRange);
    CHECK(a.blast_strength == initial_range + 1);

    CHECK(!a.can_kick);
    a.pick_up(Item::Kick);
    CHECK(a.can_kick);
}

TEST_CASE("Agent reset restores defaults", "[agent]") {
    Agent a;
    a.ammo = 5; a.blast_strength = 10; a.can_kick = true; a.is_alive = false;
    a.reset();
    CHECK(a.ammo == 1);
    CHECK(a.blast_strength == DEFAULT_BLAST_STRENGTH);
    CHECK(!a.can_kick);
    CHECK(a.is_alive);
}

TEST_CASE("Flame ticks and dies", "[flame]") {
    Flame f;
    f.life = 2;
    CHECK(!f.is_dead());
    f.tick(); CHECK(!f.is_dead());
    f.tick(); CHECK(f.is_dead());
}
