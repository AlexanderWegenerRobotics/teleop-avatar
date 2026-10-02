#include "sim_env/wrench_truth.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// mj_contactForce acts on body(geom2), reaction on body(geom1); +1 verified against qfrc_constraint
constexpr double kContactSignBody2 = +1.0;

template <typename T, std::size_t N>
void fillNaN(std::array<T, N>& a) { a.fill(kNaN); }

std::string fmt(double v) {
    if (std::isnan(v)) return "";
    std::ostringstream ss;
    ss.precision(6);
    ss << std::fixed << v;
    return ss.str();
}

}  // namespace

std::string wrenchTruthHeader() {
    std::ostringstream h;
    h << "sim_time;device";
    for (const char* p : {"tau_contact", "tau_friction", "tau_limit", "tau_constraint"})
        for (int i = 0; i < 7; ++i) h << ";" << p << "_" << i;
    for (const char* p : {"F_world", "F_base"})
        for (const char* a : {"fx", "fy", "fz", "mx", "my", "mz"}) h << ";" << p << "_" << a;
    h << ";ref_px;ref_py;ref_pz";
    h << ";ee_px_base;ee_py_base;ee_pz_base";
    h << ";ee_qw_base;ee_qx_base;ee_qy_base;ee_qz_base";
    h << ";ncon_arm;cart_valid\n";
    return h.str();
}

std::string wrenchTruthRow(const WrenchTruthEntry& e) {
    std::ostringstream r;
    r << fmt(e.sim_time) << ";" << e.device;
    for (const auto* v : {&e.tau_contact, &e.tau_friction, &e.tau_limit, &e.tau_constraint})
        for (int i = 0; i < 7; ++i) r << ";" << fmt((*v)[i]);
    for (const auto* v : {&e.F_world, &e.F_base})
        for (int i = 0; i < 6; ++i) r << ";" << fmt((*v)[i]);
    for (int i = 0; i < 3; ++i) r << ";" << fmt(e.ref_point_world[i]);
    for (int i = 0; i < 3; ++i) r << ";" << fmt(e.ee_pos_base[i]);
    for (int i = 0; i < 4; ++i) r << ";" << fmt(e.ee_quat_base[i]);
    r << ";" << e.ncon_arm << ";" << (e.cart_valid ? 1 : 0) << "\n";
    return r.str();
}

std::unique_ptr<WrenchTruth> WrenchTruth::create(
    const mjModel* m,
    const YAML::Node& sim_config,
    const YAML::Node& robot_config,
    const std::unordered_map<std::string, std::vector<int>>& joint_ids,
    const std::string& session_id) {

    if (!m) return nullptr;
    if (!sim_config["simulation"] || !sim_config["simulation"]["ground_truth_wrench"])
        return nullptr;

    const YAML::Node gt = sim_config["simulation"]["ground_truth_wrench"];
    if (gt["enabled"] && !gt["enabled"].as<bool>()) {
        std::cout << "[SIM] ground_truth_wrench: disabled in config." << std::endl;
        return nullptr;
    }
    if (!gt["devices"] || gt["devices"].size() == 0) {
        std::cout << "[SIM-WARN] ground_truth_wrench: enabled but no devices listed - off."
                  << std::endl;
        return nullptr;
    }

    const double rate_hz = gt["rate_hz"] ? gt["rate_hz"].as<double>() : 200.0;
    // one file per arm, e.g. ../log/arm_left_wrench_truth.csv
    const std::string log_dir = gt["log_dir"]
        ? gt["log_dir"].as<std::string>() : std::string("../log/");

    // base_pose from robot_config is the only link between MuJoCo world and the arm base frame
    auto findRobotDevice = [&robot_config](const std::string& name) -> YAML::Node {
        if (robot_config["devices"]) {
            for (const auto& d : robot_config["devices"])
                if (d["name"] && d["name"].as<std::string>() == name) return d;
        }
        return YAML::Node();
    };

    std::vector<Arm> arms;

    for (const auto& dev_node : gt["devices"]) {
        const std::string name = dev_node["name"]
            ? dev_node["name"].as<std::string>()
            : (dev_node.IsScalar() ? dev_node.as<std::string>() : std::string());
        if (name.empty()) continue;

        auto it = joint_ids.find(name);
        if (it == joint_ids.end() || it->second.empty()) {
            std::cout << "[SIM-WARN] ground_truth_wrench: device '" << name
                      << "' has no joints in the model - skipped." << std::endl;
            continue;
        }

        Arm a;
        a.name = name;
        a.jnt  = it->second;
        for (int j : a.jnt) a.dof.push_back(m->jnt_dofadr[j]);

        // body of the first joint, everything attached downstream counts as this arm
        a.root_body = m->jnt_bodyid[a.jnt.front()];

        const YAML::Node rdev = findRobotDevice(name);
        if (rdev && rdev["base_pose"]) {
            if (rdev["base_pose"]["orientation"]) {
                const auto q = rdev["base_pose"]["orientation"].as<std::vector<double>>();
                if (q.size() == 4) {
                    Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
                    quat.normalize();
                    a.R_wb = quat.toRotationMatrix();
                }
            }
            if (rdev["base_pose"]["position"]) {
                const auto p = rdev["base_pose"]["position"].as<std::vector<double>>();
                if (p.size() == 3) a.t_wb = Eigen::Vector3d(p[0], p[1], p[2]);
            }
        } else {
            std::cout << "[SIM-WARN] ground_truth_wrench: no base_pose for '" << name
                      << "' in robot_config - base-frame columns will be world-frame."
                      << std::endl;
        }

        // optional, without it only the Cartesian columns are missing
        if (dev_node["ee_body"]) {
            const std::string ee = dev_node["ee_body"].as<std::string>();
            a.ee_body = mj_name2id(m, mjOBJ_BODY, ee.c_str());
            if (a.ee_body < 0)
                std::cout << "[SIM-WARN] ground_truth_wrench: ee_body '" << ee
                          << "' not found for '" << name
                          << "' - Cartesian columns disabled, joint space still logged."
                          << std::endl;
        } else {
            std::cout << "[SIM] ground_truth_wrench: no ee_body for '" << name
                      << "' - joint-space columns only." << std::endl;
        }

        // bad path must not crash the sim
        const std::string path = log_dir + name + "_wrench_truth.csv";
        try {
            a.logger = std::make_unique<DataLogger<WrenchTruthEntry>>(
                path, wrenchTruthHeader, wrenchTruthRow, session_id);
            a.logger->start();
            a.logger->enable(true);
        } catch (const std::exception& ex) {
            std::cout << "[SIM-WARN] ground_truth_wrench: cannot open " << path << " ("
                      << ex.what() << ") - '" << name << "' skipped." << std::endl;
            continue;
        }

        std::cout << "[SIM] ground_truth_wrench: '" << name << "' " << a.dof.size()
                  << " dofs, root body " << a.root_body << ", ee body " << a.ee_body
                  << " -> " << path << std::endl;
        arms.push_back(std::move(a));
    }

    if (arms.empty()) {
        std::cout << "[SIM-WARN] ground_truth_wrench: no usable devices - off." << std::endl;
        return nullptr;
    }

    std::cout << "[SIM] ground_truth_wrench: " << arms.size() << " arm(s) at " << rate_hz
              << " Hz" << std::endl;

    return std::unique_ptr<WrenchTruth>(new WrenchTruth(m, rate_hz, std::move(arms)));
}

WrenchTruth::WrenchTruth(const mjModel* m, double rate_hz, std::vector<Arm> arms)
    : m_(m), arms_(std::move(arms)) {
    period_ = (rate_hz > 0.0) ? 1.0 / rate_hz : 0.005;

    jacp1_.assign(3 * m_->nv, 0.0);
    jacr1_.assign(3 * m_->nv, 0.0);
    jacp2_.assign(3 * m_->nv, 0.0);
    jacr2_.assign(3 * m_->nv, 0.0);
}

WrenchTruth::~WrenchTruth() = default;

bool WrenchTruth::isInSubtree(int body, int root) const {
    while (body > 0) {
        if (body == root) return true;
        body = m_->body_parentid[body];
    }
    return body == root;
}

void WrenchTruth::restartLoggers(const std::string& folder) {
    for (Arm& a : arms_) {
        if (!a.logger) continue;
        const std::string path = folder + "/" + a.name + "_wrench_truth.csv";
        try {
            a.logger->restart(path);
        } catch (const std::exception& ex) {
            std::cout << "[SIM-WARN] ground_truth_wrench: cannot reopen " << path
                      << " (" << ex.what() << ") - '" << a.name
                      << "' keeps writing to its previous file." << std::endl;
        }
    }
}

void WrenchTruth::sample(const mjData* d) {
    if (!d) return;
    if (d->time < next_sample_time_) return;
    // resync after reset/seek
    next_sample_time_ = (d->time > next_sample_time_ + 1.0) ? d->time + period_
                                                            : next_sample_time_ + period_;

    const int nv = m_->nv;

    for (const Arm& a : arms_) {
        if (!a.logger) continue;
        WrenchTruthEntry e;
        e.sim_time = d->time;
        e.device   = a.name;

        const std::size_t n = a.dof.size();

        // qfrc_constraint, split off the friction and limit parts
        for (std::size_t i = 0; i < n && i < 7; ++i)
            e.tau_constraint[i] = d->qfrc_constraint[a.dof[i]];

        for (int k = 0; k < d->nefc; ++k) {
            const int type = d->efc_type[k];
            // single-DOF unit Jacobian rows, so contribution = efc_force
            if (type == mjCNSTR_FRICTION_DOF) {
                const int dof = d->efc_id[k];
                for (std::size_t i = 0; i < n && i < 7; ++i)
                    if (a.dof[i] == dof) e.tau_friction[i] += d->efc_force[k];
            } else if (type == mjCNSTR_LIMIT_JOINT) {
                const int jid = d->efc_id[k];
                if (jid >= 0 && jid < m_->njnt) {
                    const int dof = m_->jnt_dofadr[jid];
                    // Jacobian is -1 at upper limit, +1 at lower, so sign depends on which limit is active
                    const double mid = 0.5 * (m_->jnt_range[2 * jid] + m_->jnt_range[2 * jid + 1]);
                    const double s   = (d->qpos[m_->jnt_qposadr[jid]] > mid) ? -1.0 : 1.0;
                    for (std::size_t i = 0; i < n && i < 7; ++i)
                        if (a.dof[i] == dof) e.tau_limit[i] += s * d->efc_force[k];
                }
            }
        }

        // contacts -> joint torque and net wrench about the EE
        Eigen::Vector3d F_w = Eigen::Vector3d::Zero();
        Eigen::Vector3d M_w = Eigen::Vector3d::Zero();

        const bool want_cart = (a.ee_body >= 0);
        Eigen::Vector3d ref = Eigen::Vector3d::Zero();
        if (want_cart)
            ref = Eigen::Map<const Eigen::Vector3d>(d->xpos + 3 * a.ee_body);

        for (int c = 0; c < d->ncon; ++c) {
            const mjContact& con = d->contact[c];
            const int b1 = m_->geom_bodyid[con.geom1];
            const int b2 = m_->geom_bodyid[con.geom2];

            const bool in1 = isInSubtree(b1, a.root_body);
            const bool in2 = isInSubtree(b2, a.root_body);
            if (!in1 && !in2) continue;

            mjtNum w_contact[6] = {};
            mj_contactForce(m_, d, c, w_contact);

            // contact.frame has axes as rows, so world = frame^T * contact
            mjtNum f_w[3], t_w[3];
            mju_mulMatTVec(f_w, con.frame, w_contact,     3, 3);
            mju_mulMatTVec(t_w, con.frame, w_contact + 3, 3, 3);

            const Eigen::Vector3d fw(f_w[0], f_w[1], f_w[2]);
            const Eigen::Vector3d tw(t_w[0], t_w[1], t_w[2]);

            // difference of both bodies' Jacobians handles self-collision and world contacts
            mj_jac(m_, d, jacp1_.data(), jacr1_.data(), con.pos, b1);
            mj_jac(m_, d, jacp2_.data(), jacr2_.data(), con.pos, b2);

            for (std::size_t i = 0; i < n && i < 7; ++i) {
                const int v = a.dof[i];
                double tau = 0.0;
                for (int r = 0; r < 3; ++r) {
                    const double dJp = jacp2_[r * nv + v] - jacp1_[r * nv + v];
                    const double dJr = jacr2_[r * nv + v] - jacr1_[r * nv + v];
                    tau += dJp * fw[r] + dJr * tw[r];
                }
                e.tau_contact[i] += kContactSignBody2 * tau;
            }
            ++e.ncon_arm;

            if (want_cart && (in1 != in2)) {
                // force on the arm, self-collisions are internal and excluded here (not in joint space)
                const double s = in2 ? kContactSignBody2 : -kContactSignBody2;
                const Eigen::Vector3d p(con.pos[0], con.pos[1], con.pos[2]);
                F_w += s * fw;
                M_w += s * (tw + (p - ref).cross(fw));
            }
        }

        if (want_cart) {
            const Eigen::Vector3d F_b = a.R_wb.transpose() * F_w;
            const Eigen::Vector3d M_b = a.R_wb.transpose() * M_w;
            for (int i = 0; i < 3; ++i) {
                e.F_world[i]     = F_w[i];
                e.F_world[i + 3] = M_w[i];
                e.F_base[i]      = F_b[i];
                e.F_base[i + 3]  = M_b[i];
                e.ref_point_world[i] = ref[i];
            }

            // EE pose in base frame, cross-check against O_T_EE
            Eigen::Matrix3d R_we;
            // xmat is row-major
            for (int r = 0; r < 3; ++r)
                for (int cc = 0; cc < 3; ++cc)
                    R_we(r, cc) = d->xmat[9 * a.ee_body + 3 * r + cc];

            const Eigen::Vector3d p_be = a.R_wb.transpose() * (ref - a.t_wb);
            const Eigen::Quaterniond q_be(a.R_wb.transpose() * R_we);
            for (int i = 0; i < 3; ++i) e.ee_pos_base[i] = p_be[i];
            e.ee_quat_base = {q_be.w(), q_be.x(), q_be.y(), q_be.z()};
            e.cart_valid = true;
        } else {
            fillNaN(e.F_world);
            fillNaN(e.F_base);
            fillNaN(e.ref_point_world);
            fillNaN(e.ee_pos_base);
            fillNaN(e.ee_quat_base);
            e.cart_valid = false;
        }

        a.logger->write(e);
    }
}
