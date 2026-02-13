function robot = spinn3d_build_robot()
    robot = spinn3d_robot();     % 你已有的构造函数
    try, robot.DataFormat = 'row'; end
end
