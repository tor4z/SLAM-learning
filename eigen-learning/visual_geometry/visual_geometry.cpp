#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pangolin/pangolin.h>


struct RotationMatrix {
    Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
};

struct TranslationVector {
    Eigen::Vector3d trans = Eigen::Vector3d(0, 0, 0);
};

struct QuaternionDraw {
    Eigen::Quaterniond q;
};


std::ostream &operator<<(std::ostream &out, const RotationMatrix &r)
{
    out.setf(std::ios::fixed);
    Eigen::Matrix3d matrix = r.matrix;
    out << "=";
    out << "[" << std::setprecision(2) << matrix(0, 0) << "," << matrix(0, 1) << "," << matrix(0, 2) << "],"
        << "[" << matrix(1, 0) << "," << matrix(1, 1) << "," << matrix(1, 2) << "],"
        << "[" << matrix(2, 0) << "," << matrix(2, 1) << "," << matrix(2, 2) << "],";
    return out;
}

std::istream &operator>>(std::istream &in, const RotationMatrix &r)
{
    return in;
}


std::ostream &operator<<(std::ostream &out, const TranslationVector &t)
{
    out << "=[" << t.trans(0) << "," << t.trans(1) << "," << t.trans(2) << "]";
    return out;
}


std::istream &operator>>(std::istream &in, const TranslationVector &t)
{
    return in;
}


std::ostream &operator<<(std::ostream &out, const QuaternionDraw &q)
{
    auto c = q.q.coeffs();
    out << "=[" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "]";
    return out;
}


std::istream &operator>>(std::istream &in, const QuaternionDraw &q)
{
    return in;
}


int main(int argc, char** argv)
{
    pangolin::CreateWindowAndBind("visual geometry", 1000, 600);
    glEnable(GL_DEPTH_TEST);
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1000, 600, 420, 420, 500, 300, 0.1, 1000),
        pangolin::ModelViewLookAt(3, 3, 3, 0, 0, 0, pangolin::AxisY)
    );

    const int UI_WIDTH = 500;

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f/600.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    
    // ui
    pangolin::Var<RotationMatrix> rotation_matrix("ui.R", RotationMatrix());
    pangolin::Var<TranslationVector> translation_vector("ui.t", TranslationVector());
    pangolin::Var<TranslationVector> euler_angles("ui.rpy", TranslationVector());
    pangolin::Var<QuaternionDraw> quaternion("ui.q", QuaternionDraw());
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        pangolin::OpenGlMatrix matrix = s_cam.GetModelViewMatrix();

        RotationMatrix R;
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                R.matrix(i, j) = matrix(i, j);
            }
        }
        rotation_matrix = R;

        TranslationVector t;
        t.trans = Eigen::Vector3d(matrix(0, 3), matrix(1, 3), matrix(2, 3));
        t.trans = -R.matrix * t.trans;
        translation_vector = t;

        TranslationVector euler;
        euler.trans = R.matrix.eulerAngles(2, 1, 0);
        euler_angles = euler;

        QuaternionDraw quat;
        quat.q = Eigen::Quaterniond(R.matrix);
        quaternion = quat;

        glColor3f(1.0, 1.0, 1.0);

        pangolin::glDrawColouredCube();
        // draw the original axis
        glLineWidth(3);
        glColor3f(0.8f, 0.0f, 0.0f);
        glBegin(GL_LINE);
        glVertex3f(0, 0, 0);
        glVertex3f(10, 0, 0);
        glColor3f(0.0f, 0.8f, 0.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 10, 0);
        glColor3f(0.2f, 0.2f, 1.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 10);
        glEnd();

        pangolin::FinishFrame();
    }

    return 0;
}
